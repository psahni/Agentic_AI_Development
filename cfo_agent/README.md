* Install the libraries


```
pip show langgraph langchain-groq pandas python-dotenv
```

```
pip install langgraph-checkpoint-sqlite
```

* Test the env


```
python -c "
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
import pandas
print('All imports OK')
"
```