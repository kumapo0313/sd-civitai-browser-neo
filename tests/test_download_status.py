"""Download status snapshot and queue-chain guard.

The progress bar and the "start the next item" trigger used to travel only on the
Gradio event running download_create_thread(). When the browser's connection
dropped (screen lock, sleeping tab), the bar froze and the queue stalled. The page
now polls get_download_status() and replays the triggers it missed; these tests pin
the server side of that contract.
"""

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from test_official_tags_integration import _install_stubs  # noqa: E402


def _update(**kwargs):
    return dict(kwargs, __type__='update')


class TestDownloadStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = dict(sys.modules)
        for name in [n for n in sys.modules if n == 'scripts' or n.startswith('scripts.')]:
            del sys.modules[name]
        _install_stubs()
        # Use the real download module; with Aria2 off its import starts no process.
        del sys.modules['scripts.civitai_download']
        sys.modules['modules.shared'].opts.use_aria2 = False
        sys.modules['gradio'].update = _update
        import scripts.civitai_download as dl
        import scripts.civitai_global as gl
        cls.dl = dl
        cls.gl = gl

    @classmethod
    def tearDownClass(cls):
        sys.modules.clear()
        sys.modules.update(cls._saved_modules)

    def setUp(self):
        self._orig_run = self.dl._download_create_thread
        self.gl.download_queue = []
        self.dl._missed_trigger.clear()
        self.dl._set_dl_status(dl_id=None, state='idle', results={},
                               queue_trigger=None, download_finish=None, current_model=None)

    def tearDown(self):
        self.dl._download_create_thread = self._orig_run
        self.gl.download_queue = []

    def test_run_records_handed_off_triggers(self):
        self.dl._download_create_thread = lambda finish, trigger, progress: (
            _update(), _update(value='Card'), _update(value=finish), _update(value='55555'))

        self.dl.download_create_thread('11111', '22222', None)

        status = self.dl.get_download_status()
        self.assertEqual(status['queue_trigger'], '55555')
        self.assertEqual(status['download_finish'], '11111')
        self.assertEqual(status['current_model'], 'Card')
        self.assertFalse(status['running'])

    def test_duplicate_trigger_is_ignored_while_running(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def slow_run(finish, trigger, progress):
            calls.append(trigger)
            started.set()
            release.wait(5)
            return (_update(), _update(), _update(value=finish), _update(value=trigger))

        self.dl._download_create_thread = slow_run
        self.gl.download_queue = [{'dl_id': 7}]
        worker = threading.Thread(target=self.dl.download_create_thread, args=('1', '2', None))
        worker.start()
        self.assertTrue(started.wait(5))

        self.assertTrue(self.dl.get_download_status()['running'])
        duplicate = self.dl.download_create_thread('1', '3', None)
        self.assertTrue(all(update == _update() for update in duplicate))

        release.set()
        worker.join(5)
        self.assertEqual(calls, ['2'])

    def _run_with_dropped_trigger(self, outputs):
        """Run once, with a second trigger arriving (and turned away) mid-run."""
        def run(finish, trigger, progress):
            self.dl.download_create_thread(finish, 'late', None)
            return outputs(finish, trigger)
        self.dl._download_create_thread = run
        return self.dl.download_create_thread('1', '2', None)

    def test_dropped_trigger_keeps_the_queue_moving(self):
        # The run ends the chain (queue_trigger unchanged) although an item is queued
        # whose start trigger was turned away: it must hand out a new queue trigger.
        self.gl.download_queue = [{'dl_id': 8}]

        outputs = self._run_with_dropped_trigger(lambda finish, trigger: (
            _update(), _update(), _update(value='99999'), _update(value=trigger)))

        self.assertNotEqual(outputs[3]['value'], '2')
        self.assertEqual(outputs[2]['value'], '1')

    def test_no_retrigger_while_awaiting_a_choice(self):
        self.gl.download_queue = [{'dl_id': 9, '_awaiting_choice': True}]

        outputs = self._run_with_dropped_trigger(lambda finish, trigger: (
            _update(value='<chooser>'), _update(), _update(value=None), _update(value=None)))

        self.assertIsNone(outputs[3]['value'])
        self.assertTrue(self.dl.get_download_status()['awaiting_choice'])

    def test_no_retrigger_without_a_dropped_trigger(self):
        self.gl.download_queue = [{'dl_id': 10}]
        self.dl._download_create_thread = lambda finish, trigger, progress: (
            _update(), _update(), _update(value='99999'), _update(value=trigger))

        outputs = self.dl.download_create_thread('1', '2', None)

        self.assertEqual(outputs[3]['value'], '2')

    def test_status_lists_queue_and_results(self):
        self.gl.download_queue = [{'dl_id': 3}, {'dl_id': 4}]
        self.dl._record_dl_result(2, 'completed')
        self.dl._record_dl_result(1, 'failed')

        status = self.dl.get_download_status()

        self.assertEqual(status['queue_ids'], ['3', '4'])
        self.assertEqual(status['results'], {'2': 'completed', '1': 'failed'})
        self.assertFalse(status['awaiting_choice'])

    def test_result_history_is_bounded(self):
        for dl_id in range(self.dl._RESULT_HISTORY + 10):
            self.dl._record_dl_result(dl_id, 'completed')

        results = self.dl.get_download_status()['results']

        self.assertEqual(len(results), self.dl._RESULT_HISTORY)
        self.assertNotIn('0', results)

    def test_tracking_progress_updates_status_and_forwards(self):
        seen = []
        tracked = self.dl._tracking_progress(lambda p, desc=None: seen.append((p, desc)), 5)

        tracked(0.5, desc='Downloading: x')

        status = self.dl.get_download_status()
        self.assertEqual(status['dl_id'], '5')
        self.assertEqual(status['progress'], 0.5)
        self.assertEqual(status['desc'], 'Downloading: x')
        self.assertEqual(seen, [(0.5, 'Downloading: x')])


if __name__ == '__main__':
    unittest.main()
