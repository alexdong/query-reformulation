# loop through all json in current directory. if the json has a `type` property, continue; if it has a `instance_of` property, rename it to `type`; if it has neither, add a `type` property with the value retrieved from `ollama` API like what we did in `facts/_main.py`. ai!

