from collections import namedtuple
import toolz as tz
import yaml


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


Comment = namedtuple("Comment", "content, type, line_key")


def _clean_line(s: str):
    return s.replace("'", "").replace('"', "").strip()


def _update_fnc(append, new_data, parent):
    if isinstance(parent, list):
        if parent == new_data:  # key was missing
            return parent
        if append:
            return parent + new_data
    return new_data


def _unbalanced(s: str, qchar: str, hash_idx: int) -> bool:
    idx = [s.find(qchar, 0, hash_idx)]
    while idx[-1] >= 0:
        idx.append(s.find('"', idx[-1] + 1, hash_idx))
    return len(idx) % 2 == 0


def _has_quoted_hashtag(s: str, hash_idx: int) -> bool:
    return _unbalanced(s, '"', hash_idx) or _unbalanced(s, "'", hash_idx)


def _read_comments(filepath: str) -> list[Comment]:
    comments, in_block = [], False
    with open(filepath, "r") as fp:
        prev_line = None
        for l in fp.readlines():
            l = l.strip()
            if not l:  # blank line
                continue
            hash_idx = l.rfind("#")  # -1 if not found
            if hash_idx >= 0:
                if l.startswith("#"):  # entire line comment
                    if prev_line is None:  # comments at start of file
                        comments.append(Comment(l, "block" if in_block else "line", None))
                    else:
                        comments.append(Comment(l, "block" if in_block else "line", _clean_line(prev_line)))
                    in_block = True
                    prev_line = l
                else:
                    prev_line = l
                    in_block = False
                    if _has_quoted_hashtag(l, hash_idx):
                        print(f"WARNING: # found inside string, ignoring line")
                    else:  # inline comment
                        comments.append(Comment(l[hash_idx:], "inline", _clean_line(l[:hash_idx])))
                        prev_line = _clean_line(l[:hash_idx])
            else:
                in_block = False
                prev_line = l
    return comments


def _add_yaml_comments(filepath: str, comments: list[Comment]) -> list[str]:
    new_lines, cmnt_idx = [], 0
    with open(filepath, "r") as fp:
        prev_line = None
        for l in fp.readlines():
            cleaned = _clean_line(l)
            if cmnt_idx < len(comments):
                comment = comments[cmnt_idx]
                if comment.type == "inline":
                    if comment.line_key == cleaned:
                        l = l.rstrip() + "  " + comment.content + "\n"
                        cmnt_idx += 1

                if cmnt_idx < len(comments):
                    comment = comments[cmnt_idx]
                    if comment.line_key == prev_line:
                        new_lines.append(comment.content + "\n")
                        cmnt_idx += 1
                        while cmnt_idx < len(comments) and comments[cmnt_idx].type == "block":
                            new_lines.append(comments[cmnt_idx].content + "\n")
                            cmnt_idx += 1
                    if (
                        cmnt_idx < len(comments)
                        and comments[cmnt_idx].type == "inline"
                        and comments[cmnt_idx].line_key == cleaned
                    ):
                        l = l.rstrip() + "  " + comments[cmnt_idx].content + "\n"
                        cmnt_idx += 1

            prev_line = cleaned
            new_lines.append(l)

        with open(filepath, "w") as fp:
            fp.writelines(new_lines)


def append_to_yaml(filepath: str, key_seq: list, new_data, append: bool = True) -> None:
    """Add data to a yaml file, preserving comments in the process.
       CAVEAT - if adding to or creating a list element, be sure that new_data is itself a list

    Args:
        filepath (str): filepath to the yaml file to modify
        key_seq (list): tz.get_in() key syntax to the data being modified
        new_data (_type_): the data being added
        append (bool, optional): append to or replace value at the key_path. Defaults to True.
    """
    with open(filepath, "r") as fp:
        file_data = yaml.safe_load(fp)

    comments = _read_comments(filepath)

    # update data
    updater = tz.partial(_update_fnc, append, new_data)
    new_data = tz.update_in(file_data, key_seq, func=updater, default=new_data)

    # write updated data to file ... without comments
    with open(filepath, "w") as fp:
        yaml.dump(new_data, fp, Dumper=IndentDumper, sort_keys=False)

    # add comments back in
    _add_yaml_comments(filepath, comments)
