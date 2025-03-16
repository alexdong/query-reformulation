Once we have the knowledge graph in `../facts`, we can start generating
subqueries. 

There are 3 types of subqueries that can be generated from the facts graph. 

1. Comparison: Compare two to five entities based on a common property. 
2. Expansion: Expand a single entity into subdomains
3. Chaining: Chain two to five entities based on a common property 

Our goal is to generate 1333 subqueries of each type, this will give us 4000
subqueries. Then with 25 queries per subquery, we'll have 100k
`query===>subqueries` pairs. That's a decent training dataset for the size of
the model we are using.


1. Comparison Subqueries 
---------------------

To generate the comparison subqueries, we first locate N random entities of the
same `properties.type` or `properties.instance_of`. Then pick a random common
property and generate the subqueries. The files should be generated into
`dataset/subqueries-comparison.txt` where each line is a \n separated list of
subqueries. 

Example:

1. Pick a `type` value and find random entities: "City" -> "Dunedin",
   "Wellington", "Auckland"
2. Find a random common `properties` -> "population"
3. Subqueries -> `Dunedin population\nWellington population\nAuckland
   population`


2. Expansion Subqueries 
--------------------

To generate the expansion subqueries, we first locate one random entity with a
property that contains a CSV list of entities with at least 3 entities or
qualifiers. Then generate the subqueries based on the values in the CSV list.
The files should be generated into `dataset/subqueries-expansion.txt`. 

Example:

1. Pick a random entity with CSV property value: "Oral tradition" ->
   "properties.purpose" -> "Education, entertainment, preservation"
2. Break down the value to generate subqueries: `Oral tradition education\nOral
   tradition entertainment\nOral tradition preservation`


3. Chaining Subqueries 
-------------------

To generate the chaining subqueries, we first locate one random entity, then
use the `relationship` to traverse to an adjacent property (if it exists) and
log each traverse in the form of a "entity relationship". This process stops
when we reach a dead-end where the destination entity doesn't exist on disk.

Example:

1. Pick a random entity: "Larnach Castle" and a random relationship:
   "located_in"
2. Traverse the relationship to find the next entity: "Otago" and its type "Region". 
3. Log the subquery: `Larnach Castle Located in Region`
4. Repeat the process for "Otago" and choose a random relationship: "has_river"
   -> "Clutha River". 
5. Log the subquery: `Region has river`.
(We can repeat 2-5 to chain more entities for extra steps)
6. End with a property: `Clutha River` `properties:basin_area` -> "23,290 km2"
7. Log the final subquery: `Clutha River basin area`
8. The final subqueries is `Larnach Castle Located in Region\nRegion has a
   river\nRiver basin area`


4. Handy Utility Commands 
----------------------

CLI commands to explore the shape of data:

```zsh 
# Find all properties 
fd -e json -x jq -r '.properties | keys[]' | sort
| uniq -c | sort -nr

# Find all properties whose value contains a CSV list of entities 
fd -e json -x jq -r '.properties | to_entries[] | select(.value | contains(",")) | .key' | sort | uniq -c | sort -nr

# Find all relationships 
fd -e json -x jq -r '.relationship[] | keys[]' | sort | uniq -c | sort -nr ```

# Find all *.json that's missing a `properties.type`

```
