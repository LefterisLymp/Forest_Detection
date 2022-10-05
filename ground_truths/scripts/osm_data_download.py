# script that downloads all OSM entries for greek beaches and stores in .GeoJSON file

import os
import overpy
import shapely.geometry
import shapely.ops
import geojson
import time

DATA_DIR = '/data/data1/users/lefterislymp/ground_truths/data'

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

categories = frozenset(['forest', 'agriculture', 'residential', 'beach'])
categories2sur = {'wood': 'forest', 'meadow': 'agriculture', 'farmland': 'agriculture', 'residential': 'residential',
                  'beach': 'beach'}


# creates geoJSON file entry for a specific beach given the nodes
def appendCategory(way, nodes, features):
    surface = way.tags.get('natural') if way.tags.get('natural') == 'wood' or way.tags.get(
        'natural') == 'beach' else way.tags.get('landuse')

    surface = categories2sur[surface]

    # prepare entry
    geom = shapely.geometry.Polygon(nodes)
    tags = way.tags
    tags['wayOSMId'] = way.id
    # You can comment the next line
    tags['surface'] = surface
    entry = geojson.Feature(geometry=geom, properties=tags)

    # update features
    if surface in features:
        features[surface].append(entry)
    else:
        features[surface] = [entry]


# downloads OSM data for given query and saves them in a .geoJSON file
def saveOSMData(queries, debug=False):
    features = {}

    # query overpass
    api = overpy.Overpass()

    relations = []
    ways = []

    for query in queries:
        res = api.query(query)
        print("Sleeping...")
        time.sleep(30)  # In order not to have too many OSM requests per minute
        relations += res.relations
        ways += res.ways

    # for each relation in the result set, we must
    # parse all the ways that it includes and merge
    # them appropriately
    for rel in relations:

        # parse ways list to merge lines
        lines = []
        for way in rel.members:
            lines.append(shapely.geometry.LineString([(node.lon, node.lat) for node in way.resolve().nodes]))
        # merge lines to create ring
        ring = shapely.ops.linemerge(lines)
        if ring.geom_type == 'MultiLineString':
            if debug:
                print('NON-CONTINUOUS BEACHES ARE NOT ACCEPTED :', rel.id)
            continue

        # append beach
        appendCategory(rel, ring.coords, features)

    # for each way in the result set, we must
    # include a beach
    for way in ways:

        # exclude the ways included in relations
        """
        if way.tags.get('natural') != 'beach':
            if debug:
                print('NOT A BEACH :', way.id)
            continue
        """
        # parse nodes
        rawNodes = [(node.lon, node.lat) for node in way.nodes]

        # append beach
        try:
            appendCategory(way, rawNodes, features)
        except:
            pass

    # create .geoJSON files
    for surface in features:
        outputFile = os.path.join(DATA_DIR, 'osm', surface + '.GeoJSON')
        with open(outputFile, "wt") as text_file:
            try:
                featureC = geojson.FeatureCollection(features[surface])
                text_file.write(geojson.dumps(featureC))
            except Exception as e:
                print(e)

    print({s: len(b) for (s, b) in features.items()})


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

beachQuery = """
[out:json][timeout:1000];
area["name:en"="Greece"]->.searchArea;          // fetch area “Greece” to search in
(                                               // gather results
  way["natural"="beach"](area.searchArea);      // query part for: “natural=beach”
  relation["natural"="beach"](area.searchArea); // (only ways and relations, not nodes)
);
out body;                                       // print results
>;
out body;
"""

agricultureQuery = """
[out:json][timeout:1000];
area["name:en"="Peloponnese"]->.searchArea;          // fetch area “Greece” to search in
(                                               // gather results
  way["landuse"="meadow"](area.searchArea);      // query part for: “natural=meadow”
  relation["landuse"="meadow"](area.searchArea); // (only ways and relations, not nodes)
  way["landuse"="farmland"](area.searchArea);      // query part for: “natural=farmland”
  relation["landuse"="farmland"](area.searchArea); // (only ways and relations, not nodes)
  );
out body;                                       // print results
>;
out body;
"""

forestQuery = """
[out:json][timeout:1000];
area["name:en"="Peloponnese"]->.searchArea;          // fetch area “Greece” to search in
(                                               // gather results
  way["natural"="wood"](area.searchArea);      // query part for: “natural=beach”
  relation["natural"="wood"](area.searchArea); // (only ways and relations, not nodes)
  );
out body;                                       // print results
>;
out body;
"""

residentialQuery = """
area["name:en"="Peloponnese"]->.searchArea;          // fetch area “Greece” to search in
(                                               // gather results
  way["landuse"="residential"](area.searchArea);      // query part for: “natural=beach”
  relation["landuse"="residential"](area.searchArea); // (only ways and relations, not nodes)
  );
out body;                                       // print results
>;
out body;
"""


def main(class_name):
    query_list = []
    if class_name == 'forest':
        query_list = [forestQuery]
    elif class_name == 'residential':
        query_list = [residentialQuery]
    elif class_name == 'agriculture':
        query_list = [agricultureQuery]
    elif class_name == 'beach':
        query_list = [beachQuery]
    elif class_name == 'all':
        query_list = [forestQuery, residentialQuery, agricultureQuery, beachQuery]
    else:
        raise Exception("Unrecognized class")

    saveOSMData(query_list)


if __name__ == "__main__":
    main('forest')
