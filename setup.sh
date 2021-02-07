# Setup.sh
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"1181301938@student.mmu.edu.my\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml