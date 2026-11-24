from pathlib import Path
import os
B=Path(__file__).resolve().parent;D=B.parent
s=(D/'fill_form.py').read_text(encoding='utf-8-sig').replace("D = os.path.dirname(os.path.abspath(__file__))","D = str(B.parent)")
s=s.replace("word.Visible = False","print('Word created',flush=True)\nword.Visible = False")
s=s.replace("    doc.SaveAs2(OUT_DOC, 0)","    print('Source opened',flush=True)\n    doc.SaveAs2(OUT_DOC, 0)")
s=s.replace("    t = doc.Tables(1)","    print('Saved initial copy',flush=True)\n    t = doc.Tables(1)")
s=s.replace("    fill_cell(doc, t.Cell(11, 1)","    print('Filling first',flush=True)\n    fill_cell(doc, t.Cell(11, 1)")
s=s.replace("    fill_cell(doc, t.Cell(12, 1)","    print('Filling second',flush=True)\n    fill_cell(doc, t.Cell(12, 1)")
s=s.replace("    rows = schedule_rows", "    print('Filling schedule',flush=True)\n    rows = schedule_rows")
s=s.replace("        t.Rows.Add", "        print('Adding row',n_slots,flush=True)\n        t.Rows.Add")
s=s.replace("    doc.Save()", "    print('Final save',flush=True)\n    doc.Save()")
s=s.replace("    while len(rows) > n_slots:\n        print('Adding row',n_slots,flush=True)\n        t.Rows.Add(t.Rows(15 + n_slots))\n        n_slots += 1", "    if len(rows) > n_slots:\n        t.Rows(19).Select()\n        word.Selection.InsertRowsAbove(len(rows)-n_slots)\n        print('Inserted schedule rows',flush=True)")
exec(compile(s,str(D/'fill_form.py'),'exec'),globals())

