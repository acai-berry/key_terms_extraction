# key_terms_extraction
Natural Language Processing project.
Given the set of articles stored in XML file, the purpose of the project is to explore the content of those articles by finding the keywords which are the most relevant per each article. The project includes parsing XML file, preprocessing the texts (tokenizing, lemmatizing, getting rid of not relevant content), using Sklearn TF-IDF metric to find the most relevant keywords for each article (those which can represent articles' content) and presenting the results in readable format.

Tasks*:
1. Reading an XML-file containing stories and headlines.
2. Extracting the headers and the text.
3. Tokenizing each text.
4. Lemmatizing each word in the story.
5. Getting rid of punctuation, stopwords, and non-nouns with the help of NLTK.
6. Counting the TF-IDF metric for each word in all stories.
7. Picking the five best scoring words.
8. Printing each story's headline and the five most frequent words in descending order in a given format.

Technologies used:
- Python
- Sklearn (TfidfVectorizer) 
- NLTK
- Pandas
- BeautifulSoup


*Project based on tasks from JetBrains Academy. 
