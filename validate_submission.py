import csv
import re
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

def validate_submission(csv_path):
    errors = []
    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return ["Row 1 must be the header row; file is empty."]
            if header != REQUIRED_HEADER:
                return [f"Row 1 (header) must be exactly: {','.join(REQUIRED_HEADER)}"]
            seen_ranks = set()
            by_rank = []
            for row_num, row in enumerate(reader, start=2):
                if not row: continue
                if len(row) < 4:
                    errors.append(f"Row {row_num}: Must have 4 elements.")
                    continue
                cid, rank_s, score_s = row[0], row[1], row[2]
                if not CANDIDATE_ID_PATTERN.match(cid):
                    errors.append(f"Row {row_num}: Invalid candidate_id '{cid}'.")
                try:
                    rank = int(rank_s)
                    seen_ranks.add(rank)
                except ValueError:
                    errors.append(f"Row {row_num}: rank must be an integer.")
                    rank = None
                try:
                    score = float(score_s)
                except ValueError:
                    errors.append(f"Row {row_num}: score must be a float.")
                    score = None
                if rank is not None and score is not None:
                    by_rank.append((rank, score, cid))
            missing = set(range(1, 101)) - seen_ranks
            if missing:
                errors.append(f"Missing ranks: {sorted(missing)}")
            by_rank.sort(key=lambda x: x[0])
            for i in range(len(by_rank) - 1):
                if by_rank[i][1] < by_rank[i+1][1]:
                    errors.append(f"Score violation: rank {by_rank[i][0]} < rank {by_rank[i+1][0]}.")
                if by_rank[i][1] == by_rank[i+1][1] and by_rank[i][2] > by_rank[i+1][2]:
                    errors.append(f"Tie-breaker error at ranks {by_rank[i][0]} and {by_rank[i+1][0]}.")
    except Exception as e:
        errors.append(f"Could not read file: {e}")
    return errors

if __name__ == '__main__':
    issues = validate_submission('submission.csv')
    if not issues:
        print("🎉 SUCCESS: Your submission file is flawless and perfectly formatted!")
    else:
        print("❌ VALIDATION FAILED:")
        for err in issues: print(f"  - {err}")