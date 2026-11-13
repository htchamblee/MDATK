"""GTK test of RNAdo's Cluster profiles window (run under xvfb-run -a): a new profile edited, a similar cluster made
by Duplicate, queues edited, run speeds learnt from a folder, saved per user (and on close), and the names checked
before saving."""
import os
import time

import pytest

gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
except (ValueError, ImportError):  # pragma: no cover
    pytest.skip("GTK 3 introspection data not installed", allow_module_level=True)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    pytest.skip("no display (run under xvfb-run)", allow_module_level=True)
if not Gtk.init_check([])[0]:  # pragma: no cover
    pytest.skip("cannot initialise GTK", allow_module_level=True)

from mdatk import clusters  # noqa: E402
from mdatk.gui.clusterdialog import ClusterProfilesWindow  # noqa: E402
from test_cluster_profiles import pmemd_out  # noqa: E402


def pump(seconds=0.1):
    end = time.time() + seconds
    while time.time() < end:
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        time.sleep(0.005)


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    w = ClusterProfilesWindow()
    w.show_all()
    pump()
    yield w
    w.destroy()
    pump()


def test_new_duplicate_edit_learn_and_save(window, tmp_path):
    w = window
    assert w.profiles == [] and not w.editor.get_sensitive()
    w.new_profile()
    pump()
    assert w.current == 0 and w.editor.get_sensitive()
    w.name.set_text("Grace")
    w.cushion.set_value(12)
    w.header.get_buffer().set_text("#SBATCH --gres=gpu:a100:1\nmodule load Amber/24")
    w.queues.clear()
    w.queues.append(["gpu", 48.0, "#SBATCH --partition=gpu"])
    w.queues.append(["long", 168.0, "#SBATCH --partition=long; #SBATCH --qos=long"])
    runs = tmp_path / "runs"
    runs.mkdir()
    pmemd_out(runs / "prod-001.out", natom=52345, s_per_ns=159.73)
    assert len(w.learn_from_folder(str(runs))) == 1
    w.duplicate()  # FASTER: very similar, not identical
    pump()
    assert w.current == 1 and w.name.get_text() == "Grace copy"
    w.name.set_text("FASTER")
    w.gpu.set_text("A40")
    w.header.get_buffer().set_text("#SBATCH --gres=gpu:a40:1\nmodule load Amber/24")
    assert w.save()
    back = clusters.load_profiles()
    assert [p.name for p in back] == ["Grace", "FASTER"]
    grace, faster = back
    assert grace.cushion == 12 and grace.gpu == "A100" and grace.header.startswith("#SBATCH --gres=gpu:a100:1")
    assert [(q.name, q.max_hours, q.directives) for q in grace.queues] == [
        ("gpu", 48.0, "#SBATCH --partition=gpu"), ("long", 168.0, "#SBATCH --partition=long\n#SBATCH --qos=long")]
    assert [(b.atoms, b.s_per_ns) for b in grace.benchmarks] == [(52345, 159.73)]
    assert faster.gpu == "A40" and faster.queues == grace.queues and faster.benchmarks == grace.benchmarks
    assert faster.speeds() == []  # the A100 runs do not count on the A40
    # selecting the other profile shows its values
    w.listbox.select_row(w.listbox.get_row_at_index(0))
    pump()
    assert w.name.get_text() == "Grace" and len(w.speeds) == 1


def test_names_are_checked_and_closing_saves(window):
    w = window
    w.new_profile()
    w.new_profile()
    pump()
    w.name.set_text("Grace")
    w.listbox.select_row(w.listbox.get_row_at_index(0))
    pump()
    w.name.set_text("Grace")
    assert not w.save() and "names used twice: Grace" in w.status.get_text()
    assert clusters.load_profiles() == []
    w.name.set_text("FASTER")
    assert w._on_delete() is False  # closing saves
    assert [p.name for p in clusters.load_profiles()] == ["FASTER", "Grace"]
    w.delete()
    pump()
    assert w.save() and [p.name for p in clusters.load_profiles()] == ["Grace"]
