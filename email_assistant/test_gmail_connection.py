from gmail_tools import search_emails

# Try fetching your 3 most recent emails
result = search_emails("in:inbox", max_results=3)
print(result)

