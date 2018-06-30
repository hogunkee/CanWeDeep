from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir':'data2'})
google_crawler.crawl(keyword='light makeup before after', max_num=1000)
