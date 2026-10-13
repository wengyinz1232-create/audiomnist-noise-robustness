"""Build a code-only archive with an allowlist. Never submit automatically."""
import zipfile
from common import ROOT


def main():
    files=list(ROOT.glob('*.py'))+[ROOT/'requirements.txt',ROOT/'requirements-plots.txt',ROOT/'Readme.txt',ROOT/'README.md',
        ROOT/'METHODS_AND_SOURCES.md',ROOT/'REQUIREMENTS_CHECKLIST.md',ROOT/'rf_settings.json']
    destination=ROOT/'code_only_no_data.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.name)
    with zipfile.ZipFile(destination) as archive:
        assert all('/' not in n and n.endswith(('.py','.txt','.md','.json')) for n in archive.namelist())
        assert 'rf_settings.json' in archive.namelist()
        assert 'run_pipeline.py' in archive.namelist() and 'Readme.txt' in archive.namelist()
    print('CODE_ONLY_ZIP',destination,'files',len(files))


if __name__=='__main__':main()
