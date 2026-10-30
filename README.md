```c
    /\_/\           ___
   = o_o =_______    \ \  -beyond the arc-
    __^      __(  \.__) )
(@)<_____>__(_____)____/
```

~ a data + ML project

viewable [here](https://kr1s7on-beyond-the-arc-app-g7kvhn.streamlit.app/)

note
```bash
pip install -r requirements.txt
streamlit run app.py
```

it aims to cover the following:

### mvp (done)

    - clean and prep shot data
    - visualize basic hot zones
    - build a basic shot success model (Logistic Regression or Random Forest)
    - share visuals and code cleanly (GitHub repo, notebooks)

### more depth (done)

once MVP is done:

    - add features like:
    - Shot angle, quarter, time left
    - Player ID (group patterns)
    - Tune model, try XGBoost
    - Add interactivity to visuals (e.g., with Tableau or Streamlit)

### exploratory (not done)

if i have time:

    - context features (e.g., dribbles, defender distance)
    - deep learning (like using player movement)
    - animate shot charts (motion link)
    - dashboards or story presentations
