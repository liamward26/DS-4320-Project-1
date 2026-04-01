# DS 4320 Project 1 Pipeline: Do Demographics Improve Movie Recommendation Systems?

This notebook runs the full project pipeline in one place:
1. prepare the database in DuckDB
2. query and engineer analysis features
3. fit baseline and enhanced regression models
4. visualize the final comparison

## 1. Imports and Setup

The notebook uses the refactored pipeline scripts so the same logic can be reused in both batch files and notebook cells.


```python
from pathlib import Path
import importlib.util

BASE = Path(".")

def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

prep = load_module("prep_module", str(BASE / "01_prep_data_refactored.py"))
analysis = load_module("analysis_module", str(BASE / "02_analysis_refactored.py"))
viz = load_module("viz_module", str(BASE / "03_visualization_refactored.py"))
```

## 2. Data Preparation

This step downloads MovieLens, creates synthetic demographics, injects a small controlled demographic signal into ratings, and loads the final tables into DuckDB.


```python
prep_output = prep.run_prep(cleanup=False, export_parquet_files=True)
prep_output
```




    {'db_path': 'data/movielens.db',
     'table_counts': {'ratings': 32000204,
      'movies': 87585,
      'tags': 2000072,
      'links': 87585,
      'users_demo': 200948},
     'data_dir': 'data'}



## 3. Validate the DuckDB database

These quick checks show that the relational database was created successfully.


```python
import duckdb

con = duckdb.connect(prep_output["db_path"])
con.execute("SHOW TABLES").fetchdf()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>links</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movies</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ratings</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tags</td>
    </tr>
    <tr>
      <th>4</th>
      <td>users_demo</td>
    </tr>
  </tbody>
</table>
</div>




```python
for table in ["ratings", "movies", "tags", "links", "users_demo"]:
    print(table)
    display(con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf())
```

    ratings



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17</td>
      <td>4.300013</td>
      <td>944249077</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>25</td>
      <td>1.421444</td>
      <td>944250228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>29</td>
      <td>2.367940</td>
      <td>943230976</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>30</td>
      <td>5.000000</td>
      <td>944249077</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>32</td>
      <td>4.904073</td>
      <td>943228858</td>
    </tr>
  </tbody>
</table>
</div>


    movies



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>


    tags



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>26479</td>
      <td>Kevin Kline</td>
      <td>1583038886</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>79592</td>
      <td>misogyny</td>
      <td>1581476297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>247150</td>
      <td>acrophobia</td>
      <td>1622483469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34</td>
      <td>2174</td>
      <td>music</td>
      <td>1249808064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34</td>
      <td>2174</td>
      <td>weird</td>
      <td>1249808102</td>
    </tr>
  </tbody>
</table>
</div>


    links



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>imdbId</th>
      <th>tmdbId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0114709</td>
      <td>862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0113497</td>
      <td>8844</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0113228</td>
      <td>15602</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0114885</td>
      <td>31357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0113041</td>
      <td>11862</td>
    </tr>
  </tbody>
</table>
</div>


    users_demo



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>age_group</th>
      <th>sex</th>
      <th>synth_source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>35-44</td>
      <td>Female</td>
      <td>ACS_sampled</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>25-34</td>
      <td>Male</td>
      <td>ACS_sampled</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25-34</td>
      <td>Female</td>
      <td>ACS_sampled</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>25-34</td>
      <td>Female</td>
      <td>ACS_sampled</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35-44</td>
      <td>Male</td>
      <td>ACS_sampled</td>
    </tr>
  </tbody>
</table>
</div>


## 4. Query Preparation

This query joins the behavioral tables and the synthetic demographic table to create the analysis dataset.


```python
query = '''
SELECT
    r.userId,
    r.movieId,
    r.rating,
    r.timestamp,
    m.genres,
    u.age_group,
    u.sex
FROM ratings r
JOIN movies m ON r.movieId = m.movieId
JOIN users_demo u ON r.userId = u.userId
'''
print(query)
joined_preview = con.execute(query + " LIMIT 5").fetchdf()
joined_preview
```

    
    SELECT
        r.userId,
        r.movieId,
        r.rating,
        r.timestamp,
        m.genres,
        u.age_group,
        u.sex
    FROM ratings r
    JOIN movies m ON r.movieId = m.movieId
    JOIN users_demo u ON r.userId = u.userId
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>genres</th>
      <th>age_group</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17</td>
      <td>4.300013</td>
      <td>944249077</td>
      <td>Drama|Romance</td>
      <td>35-44</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>25</td>
      <td>1.421444</td>
      <td>944250228</td>
      <td>Drama|Romance</td>
      <td>35-44</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>29</td>
      <td>2.367940</td>
      <td>943230976</td>
      <td>Adventure|Drama|Fantasy|Mystery|Sci-Fi</td>
      <td>35-44</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>30</td>
      <td>5.000000</td>
      <td>944249077</td>
      <td>Crime|Drama</td>
      <td>35-44</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>32</td>
      <td>4.904073</td>
      <td>943228858</td>
      <td>Mystery|Sci-Fi|Thriller</td>
      <td>35-44</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Analysis Rationale

The baseline model uses only behavioral data, while the enhanced model adds age group and sex. Ridge regression is used because the project predicts a continuous rating and includes many correlated features after genre expansion. Continuous numeric features are standardized, binary genre columns are passed through directly, and categorical demographics are one-hot encoded.

## 6. Solution Analysis

Run the modeling pipeline and compare the baseline and enhanced models using RMSE, MAE, and R².


```python
analysis_output = analysis.run_analysis()
analysis_output["results"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>baseline</td>
      <td>0.846442</td>
      <td>0.646342</td>
      <td>0.345447</td>
    </tr>
    <tr>
      <th>1</th>
      <td>enhanced</td>
      <td>0.846440</td>
      <td>0.646341</td>
      <td>0.345450</td>
    </tr>
  </tbody>
</table>
</div>




```python
analysis_output["baseline_coefficients"].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>user_mean_rating</td>
      <td>0.416810</td>
    </tr>
    <tr>
      <th>0</th>
      <td>movie_mean_rating</td>
      <td>0.396599</td>
    </tr>
    <tr>
      <th>11</th>
      <td>genre_Documentary</td>
      <td>0.052983</td>
    </tr>
    <tr>
      <th>16</th>
      <td>genre_IMAX</td>
      <td>-0.051963</td>
    </tr>
    <tr>
      <th>3</th>
      <td>user_rating_count</td>
      <td>0.029981</td>
    </tr>
    <tr>
      <th>15</th>
      <td>genre_Horror</td>
      <td>0.021353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie_rating_count</td>
      <td>-0.020358</td>
    </tr>
    <tr>
      <th>12</th>
      <td>genre_Drama</td>
      <td>0.017648</td>
    </tr>
    <tr>
      <th>17</th>
      <td>genre_Musical</td>
      <td>0.013010</td>
    </tr>
    <tr>
      <th>20</th>
      <td>genre_Sci-Fi</td>
      <td>-0.009924</td>
    </tr>
  </tbody>
</table>
</div>




```python
analysis_output["enhanced_coefficients"].head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>user_mean_rating</td>
      <td>0.416726</td>
    </tr>
    <tr>
      <th>0</th>
      <td>movie_mean_rating</td>
      <td>0.396611</td>
    </tr>
    <tr>
      <th>11</th>
      <td>genre_Documentary</td>
      <td>0.052965</td>
    </tr>
    <tr>
      <th>16</th>
      <td>genre_IMAX</td>
      <td>-0.051946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>user_rating_count</td>
      <td>0.030021</td>
    </tr>
    <tr>
      <th>15</th>
      <td>genre_Horror</td>
      <td>0.021352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>movie_rating_count</td>
      <td>-0.020350</td>
    </tr>
    <tr>
      <th>12</th>
      <td>genre_Drama</td>
      <td>0.017644</td>
    </tr>
    <tr>
      <th>17</th>
      <td>genre_Musical</td>
      <td>0.013008</td>
    </tr>
    <tr>
      <th>20</th>
      <td>genre_Sci-Fi</td>
      <td>-0.009917</td>
    </tr>
    <tr>
      <th>6</th>
      <td>genre_Adventure</td>
      <td>-0.009075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>genre_(no genres listed)</td>
      <td>0.008552</td>
    </tr>
    <tr>
      <th>8</th>
      <td>genre_Children</td>
      <td>-0.008541</td>
    </tr>
    <tr>
      <th>7</th>
      <td>genre_Animation</td>
      <td>0.006365</td>
    </tr>
    <tr>
      <th>23</th>
      <td>genre_Western</td>
      <td>0.005686</td>
    </tr>
  </tbody>
</table>
</div>



## 7. Visualization

The final figure focuses on RMSE because the project question is whether demographics provide a meaningful improvement in predictive accuracy. A tight y-axis range makes the negligible difference between the two models visible.


```python
import plotly.io as pio
pio.renderers.default = "browser"
fig = viz.run_visualization()
fig
```

## 8. Interpretation

The enhanced model produces virtually identical performance to the baseline model. Behavioral features such as user and movie average ratings dominate the coefficients, while demographic features contribute little to no measurable improvement in recommendation quality.
