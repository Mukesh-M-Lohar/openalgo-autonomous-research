import json
import re
import sys


def main():
    html_path = "data/runs/run_c7ddef89/dashboard.html"
    try:
        with open(html_path) as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read file: {e}")
        sys.exit(1)

    start_tag = '<script id="runs-database" type="application/json">'
    end_tag = "</script>"

    start_idx = content.find(start_tag)
    if start_idx == -1:
        print("runs-database script tag not found!")
    else:
        start_idx += len(start_tag)
        end_idx = content.find(end_tag, start_idx)
        json_str = content[start_idx:end_idx].strip()
        try:
            data = json.loads(json_str)
            print("Successfully loaded runs-database JSON! Number of runs:", len(data))
        except Exception as e:
            print("Failed to parse runs-database JSON:", e)
            print("Snippet around error:")
            err_pos = re.search(r"char (\d+)", str(e))
            if err_pos:
                pos = int(err_pos.group(1))
                print(json_str[max(0, pos - 100) : min(len(json_str), pos + 100)])

    script_start = "<script>"
    script_start_idx = content.find(script_start, start_idx)
    if script_start_idx != -1:
        script_start_idx += len(script_start)
        script_end_idx = content.find(end_tag, script_start_idx)
        js_code = content[script_start_idx:script_end_idx].strip()
        print("Successfully found JS main script block! Length:", len(js_code))
        idx = js_code.find("function selectStrategy")
        if idx != -1:
            print("\n--- selectStrategy snippet ---")
            print(js_code[idx : idx + 1500])


if __name__ == "__main__":
    main()
