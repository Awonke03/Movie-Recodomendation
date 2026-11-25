import subprocess,sys,time,urllib.request,re
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
  if page.locator('#for-you-nav').is_visible():
   page.locator('[data-view="discover"]').click();page.locator('#cards .taste-actions button').first.click();page.locator('#for-you-nav').click();expect(page.locator('#section-title')).to_have_text('For You');expect(page.locator('#cards .movie')).to_have_count(24);expect(page.locator('#profile-count')).to_contain_text('1 liked')
  # Mock provider data, not a live trailer playback test.
  live=b.new_page(viewport={'width':1440,'height':900});movie={'provider':'TMDB','tmdb_id':123456,'title':'Example movie','year':2026,'genres':['Action'],'poster':'https://image.tmdb.org/t/p/w500/example.jpg','backdrop':'https://image.tmdb.org/t/p/w1280/example.jpg'}
  live.route('**/health',lambda route:route.fulfill(json={'tmdb':True,'personalisation':False}))
  live.route(re.compile(r'/api/discover\?'),lambda route:route.fulfill(json={'items':[movie],'total':1,'pages':1,'genres':['Action'],'decades':[2020,2010]}))
  live.route('**/api/tmdb/123456',lambda route:route.fulfill(json={**movie,'trailer':'ABCDEFGHIJK','related':[]}))
  live.route('https://www.youtube-nocookie.com/**',lambda route:route.fulfill(body='mock player'))
  live.route('https://image.tmdb.org/**',lambda route:route.abort())
  live.goto('http://127.0.0.1:8050');expect(live.locator('#section-title')).to_have_text('Recent releases');live.locator('#cards .poster').first.click();expect(live.locator('.trailer-play')).to_be_visible()
  panel=live.locator('.trailer-panel').bounding_box();info=live.locator('.detail-info').bounding_box();assert panel['x']>info['x']
  live.locator('.trailer-play').click();expect(live.locator('#detail-trailer iframe')).to_have_attribute('src','https://www.youtube-nocookie.com/embed/ABCDEFGHIJK?autoplay=1&rel=0')
  live.locator('#close-detail').click();expect(live.locator('#detail-trailer iframe')).to_have_count(0)
  live.locator('#collection').select_option('classics');expect(live.locator('#section-title')).to_have_text('Classic cinema')
  live.close()
  assert not errors,errors
  b.close()
finally:p.terminate();p.wait(timeout=10)
