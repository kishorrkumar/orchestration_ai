import re

content = open('orchestration/gateway/studio_ui.py', encoding='utf-8').read()
dom_ids = set(re.findall(r'id=["\']([^"\']+)["\']', content))
js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', content))

print("DOM IDs found:", len(dom_ids))
print("JS IDs looked up:", len(js_ids))
missing = js_ids - dom_ids
print("MISSING IDS:", sorted(list(missing)))
