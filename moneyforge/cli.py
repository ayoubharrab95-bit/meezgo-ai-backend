import json
import sys
from .engine import Lead, rank_leads

def main() -> None:
    payload = json.loads(sys.stdin.read())
    leads = [Lead(**item) for item in payload["leads"]]
    print(json.dumps(rank_leads(leads), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
