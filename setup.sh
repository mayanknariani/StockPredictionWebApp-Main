mkdir -p ~/.streamlit/

echo "\
    [general]\n\
    email = \"mayank007@e.ntu.edu.sg\"\n\
    " > ~/.streamlit/credentials.toml

echo "\
    [server]\n\
    headless = true\n\
    enableCORS = false
    port = $PORT\n\
    " > ~/.streamlit/config.toml