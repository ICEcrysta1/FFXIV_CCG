"""通过 ``python -m scripts.fflogs_scraper`` 下载报告。

示例：
  python -m scripts.fflogs_scraper single --report ABC123 --fight 6 --source 10
  python -m scripts.fflogs_scraper batch -e 1079 --spec-name BlackMage --max-pages 3
  python -m scripts.fflogs_scraper encounters -z 39
"""

from .cli import main

if __name__ == "__main__":
    main()
