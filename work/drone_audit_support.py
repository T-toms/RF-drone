"""Small helpers shared by the lab-data audit scripts."""
import re


def replace_markdown_section(text, heading, body):
    """Replace only this heading's section; preserve later peer sections."""
    level = len(heading) - len(heading.lstrip('#'))
    match = re.search(r'^' + re.escape(heading) + r'\s*$', text, re.MULTILINE)
    replacement = body.strip() + '\n\n'
    if match is None:
        return text.rstrip() + '\n\n' + replacement
    next_heading = re.search(r'^#{1,' + str(level) + r'} ', text[match.end():], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[:match.start()] + replacement + text[end:]
