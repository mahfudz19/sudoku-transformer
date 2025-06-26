import re


# grid sudoku
def format_sudoku(s: list):
    rows = []
    for i in range(9):
        row = s[i * 9 : (i + 1) * 9]
        # Convert all elements to string for joining
        row = ["." if ch == 0 else str(ch) for ch in row]
        formatted_row = " | ".join(" ".join(row[j : j + 3]) for j in range(0, 9, 3))
        rows.append(formatted_row)
        if i % 3 == 2 and i != 8:
            rows.append("-" * 21)
    return rows


def strip_ansi_codes(text):
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def print_side_by_side(puzzle, predicted, ground_truth):
    puzzle_lines = format_sudoku(puzzle)
    truth_lines = format_sudoku(ground_truth)

    # Prepare predicted lines with color for wrong digits
    predicted_lines = []
    for i in range(9):
        row_pred = predicted[i * 9 : (i + 1) * 9]
        row_truth = ground_truth[i * 9 : (i + 1) * 9]
        row_str = ""
        for j in range(9):
            ch = str(row_pred[j])
            if row_pred[j] != row_truth[j]:
                # ANSI escape code for red color
                ch = f"\033[91m{ch}\033[0m"
            row_str += ch + " "
            if (j + 1) % 3 == 0 and j != 8:
                row_str += "| "
        predicted_lines.append(row_str)
        if (i + 1) % 3 == 0 and i != 8:
            predicted_lines.append("-" * 21)

    # Pad predicted lines based on visible length (excluding ANSI codes)
    padded_predicted_lines = []
    for line in predicted_lines:
        visible_len = len(strip_ansi_codes(line))
        padding = 25 - visible_len
        if padding > 0:
            line += " " * padding
        padded_predicted_lines.append(line)

    print(f"\n{'Puzzle':<25} {'Predicted':<25} {'Ground Truth':<25}")
    print("=" * 75)
    for p, pr, gt in zip(puzzle_lines, padded_predicted_lines, truth_lines):
        print(f"{p:<25} {pr} {gt:<25}")
