import subprocess,sys,time,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[1]
p=subprocess.Popen([sys.executable,'movie_recommendation.py'],cwd=root)
try:
 for _ in range(80):
  try:urllib.request.urlopen('http://127.0.0.1:8050/health',timeout=1);break
  except Exception:time.sleep(.2)
 else:raise RuntimeError('Server did not start')
 with sync_playwright() as w:
  b=w.chromium.launch();page=b.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto('http://127.0.0.1:8050');expect(page.locator('#cards .movie')).to_have_count(24)
  for width in [320,390,768,1024,1440,1920]:
   page.set_viewport_size({'width':width,'height':900});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
  page.locator('#query').fill('toy story');page.locator('#search-form button').click();expect(page.locator('#section-title')).to_have_text('Search results');expect(page.locator('#cards .movie').first).to_be_visible()
  page.locator('#cards .movie .similar').first.click();expect(page.locator('#section-title')).to_contain_text('Because you like');expect(page.locator('#cards .movie').first).to_be_visible()
  page.locator('#cards .movie .save').first.click();expect(page.locator('#watch-count')).to_have_text('1');page.locator('[data-view="watchlist"]').click();expect(page.locator('#saved-cards .movie')).to_have_count(1)
  page.locator('#saved-cards .poster').first.click();expect(page.locator('#details')).to_be_visible();expect(page.locator('#details a[href*="imdb.com"]')).to_be_visible();page.locator('#close-detail').click();expect(page.locator('#details')).to_be_hidden()
  assert not errors,errors
  b.close()
finally:p.terminate();p.wait(timeout=10)
