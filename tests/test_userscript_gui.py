"""GTK tests of custom script steps in the Project Planner and the wizard (run under xvfb-run -a)."""

import os

import test_workflow_gui as gui  # skips this module too when there is no display

main = gui.main  # the main-window fixture of the other GUI tests
pump = gui.pump


def _issues(wz, level="error"):
    return [r.issue.message for r in wz.issue_list.get_children()
            if getattr(r, "issue", None) and r.issue.level == level]


def test_custom_script_in_palette_planner_and_wizard(main, tmp_path):
    from mdatk.gui.planner import CATEGORY_COLORS
    from mdatk.workflows import tools_by_category
    assert ("Custom", ["user-script"]) in [(c, [t.name for t in ts]) for c, ts in tools_by_category()]
    assert "Custom" in CATEGORY_COLORS
    main.open_planner()
    pl = main.planner
    pump()
    node = pl.add_step("user-script", 40, 40)
    pump(0.2)
    assert pl.canvas.ports(node) == ([], [])
    pl.open_wizard(node)
    pump(0.5)
    wz = pl.wizards[node.id]
    assert wz.forms["script"].rows["script"][1].get_min_content_height() >= 200  # room to type a script
    assert any("the script is empty" in m for m in _issues(wz))
    # a declared input appears on the Files page and on the box; renaming it updates both
    wz.forms["inputs"].add("pdb")
    pump(0.6)
    assert "structure" in wz.input_widgets and [p.name for p in pl.canvas.ports(node)[0]] == ["structure"]
    wz.forms["inputs"].form.rows["name"][4]("model")
    pump(0.6)
    assert node.settings["inputs"][0]["name"] == "model"
    assert "model" in wz.input_widgets and "structure" not in wz.input_widgets
    assert [p.name for p in pl.canvas.ports(node)[0]] == ["model"]
    # an output, the script and a hand-picked input: no errors left, the run script is previewed
    wz.forms["outputs"].add("pdb")
    wz.forms["script"].rows["script"][4]('cp "$MDATK_IN_MODEL" "$MDATK_OUT_STRUCTURE"\n')
    wz.forms["script"].rows["arguments"][4]("{in:modl}")
    pump(0.6)
    assert any("{in:modl} names no declared input (declared: model)" in m for m in _issues(wz))
    wz.forms["script"].rows["arguments"][4]("{in:model} {out:structure}")
    model = tmp_path / "model.pdb"
    model.write_text("END\n")
    wz.input_widgets["model"].entry.set_text(str(model))
    pump(0.6)
    assert _issues(wz) == [] and wz.write_btn.get_sensitive() and wz.run_btn.get_sensitive()
    assert [p.name for p in pl.canvas.ports(node)[1]] == ["structure"]
    buf = wz.preview.get_buffer()
    text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
    assert "export MDATK_IN_MODEL=" in text and "did not write" in text
    wz.write_files()
    pump(0.3)
    folder = os.path.join(main.project.workflow_dir, "01_custom")
    assert sorted(os.listdir(folder)) == sorted([f.name for f in wz.files])
