import requests
from lxml import html

def parse_aircraft_page(typecode):
        
    url = f"https://contentzone.eurocontrol.int/aircraftperformance/details.aspx?ICAO={typecode}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to load {url}")
        return None

    tree = html.fromstring(response.content)
    data = {'typecode': typecode}

    # Find all sections inside collapsible-content
    sections = tree.xpath('//div[contains(@class, "collapsible-content")]/h4')

    for h4 in sections:
        section_title = h4.text_content().strip()
        section_key = section_title.lower().replace(" ", "_")

        # Get all <p> siblings until the next <h4>
        parent = h4.getparent()
        idx = parent.index(h4)

        # Collect all <p> elements in the following sibling rows until next <h4>
        p_elements = []
        for i in range(idx + 1, len(parent)):
            el = parent[i]
            if el.tag == 'h4':
                break
            p_elements.extend(el.xpath('.//p'))

        for p in p_elements:
            label = p.xpath('.//span[contains(@class, "perf-label")]/text()')
            value = p.xpath('.//span[contains(@class, "perf-value")]/text()')
            unit = p.xpath('.//span[contains(@class, "perf-unit")]/text()')

            if label and value:
                label_text = label[0].strip()
                key = f"{section_key}_{label_text}".replace(" ", "_").replace("(", "").replace(")", "")
                data[key] = value[0].strip()
                if unit:
                    data[f"{key}_unit"] = unit[0].strip()

# ===== PART 2: Type of Aircraft, Technical and Similarity Info =====

    # Type of Aircraft Section
    type_labels = {
        'MainContent_wsTypeLabel': 'type',
        'MainContent_wsAPCLabel': 'APC',
        'MainContent_wsWTCLabel': 'WTC_class',
        'MainContent_wsRecatEULabel': 'RECAT_EU_class'
    }

    for elem_id, key in type_labels.items():
        val = tree.xpath(f'//span[@id="{elem_id}"]/text()')
        if val:
            data[key] = val[0].strip()

    # Technical Section
    tech_labels = {
        'MainContent_wsLabelWingSpan': 'wingspan_m',
        'MainContent_wsLabelLength': 'length_m',
        'MainContent_wsLabelHeight': 'height_m',
        'MainContent_wsLabelPowerPlant': 'powerplant'
    }

    for elem_id, key in tech_labels.items():
        val = tree.xpath(f'//span[@id="{elem_id}"]/text()')
        if val:
            data[key] = val[0].strip()

    # Optional: Performance similarity (currently "No data")
    similarity = tree.xpath('//h3[text()="Performance Similarity"]/following::p[@class="ap-detail-data"][1]/text()')
    if similarity:
        data['performance_similarity'] = similarity[0].strip()
    
    return data
