from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir':'data'})
google_crawler.crawl(keyword='makeup before after', max_num=1800)
