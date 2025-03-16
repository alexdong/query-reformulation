Give me facts about {{ entity }}.

Return a JSON that contains 10 properties and 10 relationships of the entity.
Make sure to always include a "type" in the `properties`. The "type"'s value should be close to the entity's type defined in DBpedia or Wikipedia.
Keep all properties value and the relationship values as concise as possible. Do not include `()`. Again, stay as close to valid values in DBpedia or Wikipedia as you can.

Here is an example of the JSON that should be returned:

```json
{
  "entity": "The Lord of the Rings",
  "properties": {
    "type": "Book Series",
    "author": "J. R. R. Tolkien",
    "genre": "High fantasy",
    "original_language": "English",
    "country_of_origin": "United Kingdom",
    "publication_date": "1954-1955",
    "number_of_books": "3",
    "setting": "Middle-earth",
    "main_theme": "Good vs. Evil, Power and Corruption, Friendship, Courage",
    "publisher": "George Allen & Unwin"
  },
  "relationship": [
    {"prequel": "The Hobbit"},
    {"sequel_material": "The Silmarillion"},
    {"film_adaptation": "The Lord of the Rings"},
    {"influenced": "Dungeons & Dragons"},
    {"character_appears": "Gandalf"},
    {"character_appears": "Frodo Baggins"},
    {"character_appears": "Aragorn"},
    {"takes_place_in": "Middle-earth"},
    {"major_event": "War of the Ring"},
    {"has_part": "The Fellowship of the Ring"}
  ]
}```
