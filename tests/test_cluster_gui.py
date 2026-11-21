"""GTK test of RNAdo's Cluster profiles window (run under xvfb-run -a): a new profile edited, a similar cluster made
by Duplicate, queues edited, run speeds learnt from a folder, saved per user (and on close, also of RNAdo), the names
checked before saving, values still being typed, and a profiles file that cannot be read or holds odd values."""
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


def edit_cell(view, row, column):
    """Open a cell of a table like a click would; returns its entry."""
    col = view.get_column(column)
    opened = []
    cell = col.get_cells()[0]
    handler = cell.connect("editing-started", lambda _c, e, _p: opened.append(e))
    view.grab_focus()
    view.set_cursor(Gtk.TreePath(row), col, True)
    end = time.time() + 5
    while not opened and time.time() < end:
        pump(0.05)
    cell.disconnect(handler)
    return opened[0]


def test_what_is_being_typed_is_stored(window):
    """A queue cell still open, a cushion typed and not entered: Save, another profile and closing the window
    take them (GTK 3 drops a cell's text when the cell loses the focus, and closing moves no focus)."""
    w = window
    w.new_profile()
    w.new_profile()
    pump()
    assert w.current == 1 and w.save()
    entry = edit_cell(w.queue_view, 0, 1)  # Max hours of the first queue, then a click on Save
    entry.set_text("72")
    w.name.grab_focus()
    pump(0.2)
    assert w.queues[0][1] == 72.0 and w.dirty
    assert w.save() and clusters.load_profiles()[1].queues[0].max_hours == 72.0
    entry = edit_cell(w.queue_view, 1, 0)  # a queue's name, then another profile is selected
    entry.set_text("week")
    w.listbox.select_row(w.listbox.get_row_at_index(0))
    pump(0.2)
    assert [q.name for q in w.profiles[1].queues] == ["gpu", "week"]
    assert [q.name for q in w.profiles[0].queues] == ["gpu", "long"] and w.current == 0
    entry = edit_cell(w.queue_view, 0, 2)  # the window is closed while a cell is open
    entry.set_text("#SBATCH --partition=a100")
    assert w._on_delete() is False
    assert clusters.load_profiles()[0].queues[0].directives == "#SBATCH --partition=a100"
    pump(0.2)
    w.cushion.grab_focus()  # a cushion typed, not entered, and the window closed
    w.cushion.set_text("15")
    assert not w.dirty and w._on_delete() is False
    saved = clusters.load_profiles()
    assert saved[0].cushion == 15 and saved[1].cushion == 10 and [q.name for q in saved[1].queues] == ["gpu", "week"]


GRACE = """[[profile]]
name = "Grace"
header = "#SBATCH --nodes=1"
gpu = "A100"

[[profile.queues]]
name = "gpu"
max_hours = 48.0
directives = "#SBATCH --partition=gpu"

[[profile.benchmarks]]
gpu = "NVIDIA A100"
atoms = 50000
dt = 0.004
s_per_ns = 300
when = "2026-01-01"
source = "/x/prod.out"
"""


@pytest.mark.parametrize("broken", [
    GRACE.replace('gpu = "A100"', 'gpu = "A100').encode(),  # a typo: not TOML
    GRACE.replace("Grace", "Gr\u00e4ce").encode("latin-1"),  # not UTF-8
    GRACE.replace("[[profile]]", "[profile]").encode(),
])
def test_a_profiles_file_that_cannot_be_read_is_not_replaced_unseen(tmp_path, monkeypatch, broken):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = clusters.profiles_path()
    os.makedirs(os.path.dirname(path))
    with open(path, "wb") as fh:
        fh.write(broken)
    w = ClusterProfilesWindow()
    try:
        w.show_all()
        pump()
        assert w.profiles == [] and "Could not read" in w.status.get_text()
        assert w._on_delete() is False  # nothing edited: closing leaves the file alone
        assert open(path, "rb").read() == broken and not os.path.exists(path + ".unreadable")
        w.new_profile()
        pump()
        assert w.save() and ".unreadable" in w.status.get_text()
        assert open(path + ".unreadable", "rb").read() == broken  # the user's profiles are still there
        assert [p.name for p in clusters.load_profiles()] == ["New cluster"]
        assert w.save() and ".unreadable" not in w.status.get_text()
    finally:
        w.destroy()
        pump()


def test_a_profiles_file_edited_by_hand(tmp_path, monkeypatch):
    """Numbers written as text or as integers, text written as a number: the window shows the profile (it raised
    in its handlers, or did not open), and a run speed written as an integer can be removed."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = clusters.profiles_path()
    os.makedirs(os.path.dirname(path))
    text = (GRACE.replace('name = "Grace"', "name = 5").replace("max_hours = 48.0", 'max_hours = "72"')
            .replace('header = "#SBATCH --nodes=1"', "header = 7")
            + '\n[[profile.queues]]\nname = "long"\nmax_hours = "a week"\n'
            + '\n[[profile.benchmarks]]\ngpu = "NVIDIA A100"\natoms = "many"\ndt = 0.004\ns_per_ns = 200.0\n')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    w = ClusterProfilesWindow()
    try:
        w.show_all()
        pump()
        assert not w.unreadable and w.current == 0 and w.name.get_text() == "5"
        assert [tuple(r) for r in w.queues] == [("gpu", 72.0, "#SBATCH --partition=gpu"), ("long", 48.0, "")]
        assert [(r[2], r[4]) for r in w.speeds] == [(50000, 300.0)]
        w.speed_view.get_selection().select_all()
        w.remove_speeds()
        assert len(w.speeds) == 0 and w.profiles[0].benchmarks == [] and w.dirty
    finally:
        w.destroy()
        pump()


def test_closing_rnado_saves_the_cluster_profiles(tmp_path, monkeypatch):
    from mdatk.gui.rnado import RNAdoWindow

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    app = Gtk.Application(application_id=None)
    app.register(None)
    win = RNAdoWindow(app, None, None)
    try:
        win.show_all()
        win.open_cluster_profiles()
        pump(0.3)
        w = win.cluster_profiles
        w.new_profile()
        w.new_profile()
        pump()
        w.name.set_text("New cluster")  # the name of the other profile: cannot be saved
        assert win._on_delete() is True and win.cluster_profiles is w and "names used twice" in w.status.get_text()
        assert clusters.load_profiles() == []
        w.name.set_text("Grace")
        assert win._on_delete() is False  # RNAdo closes: the profiles are saved first
        pump()
        assert [p.name for p in clusters.load_profiles()] == ["New cluster", "Grace"] and win.cluster_profiles is None
    finally:
        win.destroy()
        pump()


def test_the_window_fits_a_small_screen_whatever_the_profile_holds(window):
    """A long line of the job header or of a queue's directives scrolls: it made the window (and the place of
    Save and Close) wider than the screen; the window was taller than a 768-pixel screen."""
    w = window
    w.new_profile()
    pump()
    w.header.get_buffer().set_text("#SBATCH --nodes=1\nexport PATH=" + "/opt/amber24/bin:" * 40 + "$PATH\n"
                                   + "\n".join(f"module load tool{k}" for k in range(40)))
    w.queues.append(["all", 24.0, "; ".join(f"#SBATCH --constraint=feature{k}" for k in range(30))])
    for k in range(12):
        w.queues.append([f"q{k}", 1.0, ""])
    pump(0.3)
    minimum, _natural = w.get_preferred_size()
    assert minimum.width <= 1000 and minimum.height <= 720, (minimum.width, minimum.height)
