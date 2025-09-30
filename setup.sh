#!/bin/bash

# Create streamlit configuration directory
mkdir -p ~/.streamlit/

# Create credentials file
echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

# Create config file
echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor = '#1E40AF'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F3F4F6'\n\
textColor = '#1F2937'\n\
font = 'sans serif'\n\
" > ~/.streamlit/config.toml

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
