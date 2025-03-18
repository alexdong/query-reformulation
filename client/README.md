- [[`chat.py`]]: contains the code to interact with the model in chat mode. It
  takes in a query and returns the reformulated query as well as the time taken
  to generate the reformulation.

- [[`api.py`]]: adds an API layer on top of the model and upon serving the
  request and respond with the latency in the HTTP response header.

