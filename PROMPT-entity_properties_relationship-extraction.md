Give me facts about {% if entity %} {{ entity }} in the context of {{ related_entity }} {% else %} a random entity {% endif %} (entity as defined in WikiData or DBpedia).

Return a JSON that contains 10 properties and 10 relationships of the entity.

Here is an example of the JSON that should be returned:

```json
{
  "entity": "The Lord of the Rings",
  "properties": {
    "instance_of": "Book series",
    "author": "J. R. R. Tolkien",
    "genre": "High fantasy",
    "original_language": "English",
    "country_of_origin": "United Kingdom",
    "publication_date": "1954-1955",
    "number_of_books": "3 (though often considered as one long work)",
    "setting": "Middle-earth",
    "main_theme": "Good vs. Evil, Power and Corruption, Friendship, Courage",
     "publisher": "George Allen & Unwin"
  },
  "relationship": [
    {"prequel": "The Hobbit"},
    {"sequel_material": "The Silmarillion"},
    {"film_adaptation": "The Lord of the Rings (film series)"},
    {"influenced": "Dungeons & Dragons"},
    {"character_appears": "Gandalf"},
    {"character_appears": "Frodo Baggins"},
    {"character_appears": "Aragorn"},
    {"takes_place_in": "Middle-earth"},
    {"major_event": "War of the Ring"},
    {"has_part": "The Fellowship of the Ring"}
  ]
}```
