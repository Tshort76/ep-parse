import os
import ep_parse.epmed.core as epm


def remove_TXT_datafiles(export_dir: str, test_run=False) -> list[str]:
    """Removes any TXT files that contain signal data, since BIN files are the source of truth

    Args:
        export_dir (str): epmed export directory containing BIN files
        test_run (bool, optional): If True, returns the files that would be removed. Defaults to False.

    Returns:
        list[str]: files that were removed
    """
    to_remove = [f for f in os.listdir(export_dir) if epm.SIG_TXT_RGX.match(f)]
    if not test_run:
        for f in to_remove:
            os.remove(os.path.join(export_dir, f))

    return to_remove


def rename_BINS(export_dir: str, test_run=False) -> list[tuple[str, str]]:
    """Standardize BIN file names to remove spaces and odd characters

    Args:
        export_dir (str): epmed export directory containing BIN files
        test_run (bool, optional): If True, returns the files that would be renamed without modifying them. Defaults to False.

    Returns:
        list[tuple[str, str]]: collection of old names to new names
    """
    to_rename = []

    for f in os.listdir(export_dir):
        fnd = epm.SIG_SESSION_RGX.search(f)
        if fnd:
            k = tuple(fnd.groups())
            if k[0] == "ABL d":
                to_rename.append([f, f.replace(k[0], "ABLd", 1)])
            elif k[0] == "ABL p":
                to_rename.append([f, f.replace(k[0], "ABLp", 1)])

    if not test_run:
        for rename_pair in to_rename:
            fq_pair = list(map(lambda x: os.path.join(export_dir, x), rename_pair))
            os.rename(*fq_pair)

    return to_rename


def remove_pii(export_dir: str, case_id: str) -> None:
    """Removes any personally identifiable information from the files

    Args:
        export_dir (str): epmed export directory containing BIN files
        case_id (str): alias to use in place of the patient name
    """
    meta_files = [
        os.path.join(export_dir, f)
        for f in os.listdir(export_dir)
        if (f.lower().startswith("session") and f.lower().endswith("txt"))
    ]
    for f in meta_files:
        with open(f, "r") as fr:
            lines = fr.readlines()
        if lines[0].lower().startswith("name"):
            print(f"PII found in file: {f}")
            with open(f, "w") as fw:
                lines[1] = f"Case Id={case_id}\n"
                fw.writelines(lines[1:])


def check_partial_export(export_dir: str, assert_if=True) -> list[str]:
    """Checks to see if channels for the same Page have different sizes, which may indicate a problem

    Args:
        export_dir (str): epmed export directory containing BIN files
        assert_if (bool, optional): Whether to throw an AssertionError if a mismatch occurs. Defaults to True.

    Returns:
        list[str]: a list of files that violate the check
    """
    name_to_size = {}
    to_remove = []
    for fname in os.listdir(export_dir):
        fsize = os.path.getsize(export_dir + "/" + fname)
        f = fname.split("Page")[0]
        if f in name_to_size:
            if fsize != name_to_size[f][1]:
                _rm = fname if fsize < name_to_size[f][1] else name_to_size[f][0]
                to_remove.append(_rm)
                print(f"File {fname} has sizes {name_to_size[f][1]} and {fsize}")
        else:
            name_to_size[f] = [fname, fsize]

    if assert_if:
        assert (
            not to_remove
        ), f"Session files with varying sizes found. \n{to_remove}\n  Please run data_qa.check_partial_export(dir, False) to fix this problem."

    return to_remove


def remove_extra_page_files(directory: str, qa_check: bool = False) -> set[str]:
    """Remove redundant BIN files (extra pages) to prevent signal data duplication

    Args:
        directory (str): epmed export directory containing your BIN files
        qa_check (bool, optional): Check for equal cardinality of files per channel. Defaults to False.

    Returns:
        set[str]: The names of any files that were removed
    """
    check_partial_export(directory)
    dir_files = os.listdir(directory)

    to_keep = set()
    sig_sess = set()

    for f in dir_files:
        if f.endswith("BIN"):
            fnd = epm.SIG_SESSION_RGX.search(f)
            if fnd:
                k = tuple(fnd.groups())
                if k not in sig_sess:
                    sig_sess.add(k)
                    to_keep.add(f)
            else:
                print(f"No signal session regex match for file: {f}")
        else:
            to_keep.add(f)

    # Quality check the results
    if qa_check:
        v = {}

        for x in sig_sess:
            if x[0] in v:
                v[x[0]] += 1
            else:
                v[x[0]] = 1

        cnts = set(v.values())

        assert len(cnts) == 1, f"unequal number of sessions per channel: {v}"

    # Remove files which are not in the whitelist
    to_remove = set(dir_files) - to_keep
    for f in to_remove:  # remove old png files
        os.remove(os.path.join(directory, f))

    return to_remove


def clean_epmed_export(case_directory: str, qa_checks: bool = False):
    rename_BINS(case_directory, test_run=qa_checks)
    remove_extra_page_files(case_directory, qa_check=qa_checks)
    remove_TXT_datafiles(case_directory, test_run=qa_checks)
