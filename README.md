# DS 4320 Project 1: Demographic-Aware Content Recommendation

## Executive Summary
This project develops a relational dataset and analysis pipeline to improve content recommendations by combining user rating behavior with demographic patterns. Using the MovieLens 32M dataset and synthetic demographic data derived from U.S. Census distributions, the project explores whether incorporating age group and gender can improve recommendations for users with limited interaction history. The repository includes data construction, schema design, and a full pipeline for analysis and visualization.

## Name
Liam Ward

## NetID
vhk7vr

## DOI
tba

---

## Links (tba)

- [Press Release](./press_release.md)
- [Data Folder](./data/) this will be onedrive
- [Pipeline](./pipeline/)
- [License](./LICENSE)

---

# Problem Definition

## Initial General Problem
Recommending content (e.g., Netflix)

## Refined Specific Problem
Improve content recommendations for users with limited viewing history by incorporating demographic patterns (age group and gender) alongside past ratings.

## Rationale for Refinement
The general problem of recommending content is broad and can be approached in many ways. This project narrows the focus to users with limited interaction history, where traditional recommendation systems struggle. By applying a double diamond approach, the problem is first explored broadly and then refined to a specific, high-impact use case. Incorporating demographic patterns provides an additional signal that may improve recommendations when behavioral data is sparse.

## Motivation
Content platforms rely on recommendation systems to help users navigate large catalogs. However, these systems often perform poorly for new or inactive users due to limited data. This project is motivated by the need to improve early-stage recommendations by supplementing user behavior with demographic patterns, with the goal of increasing engagement and improving user experience.

---

# Domain Exposition

## Terminology

| Term | Definition |
|---|---|
| User | Individual interacting with the platform |
| Item | Content being recommended (e.g., movie) |
| Rating | User-provided score for an item |
| Tag | User-generated label describing content |
| Cold Start Problem | Difficulty recommending for users with little data |
| Recommendation System | System that suggests items to users |
| Demographic Features | Attributes such as age group and gender |

## Domain Description
This project operates in the domain of recommender systems, which are widely used in platforms such as streaming services and e-commerce websites. These systems analyze user interactions to suggest relevant content. A key challenge in this domain is the cold-start problem, where insufficient data about a user leads to poor recommendations. This project explores whether incorporating demographic information can improve recommendation quality in these scenarios.

---

# Data Creation

## Provenance
The dataset was obtained from the GroupLens Research MovieLens repository (https://grouplens.org/datasets/movielens/). The MovieLens 32M dataset was downloaded and extracted locally, providing multiple CSV files including ratings, movies, tags, and links. Synthetic demographic data was generated using distributions derived from U.S. Census data, assigning age group and gender to each user.

## Code Table (needs edit)

| File Name | Description | Link | 
|---|---|---|
| load_data.py | Loads MovieLens CSV files into dataframes | ./pipeline/load_data.py |
| generate_demographics.py | Generates synthetic demographic data based on Census distributions | ./pipeline/generate_demographics.py |
| preprocess.py | Cleans and prepares data for analysis | ./pipeline/preprocess.py |

## Bias Identification (edit)
Bias may arise from the MovieLens data collection process, where users are selected based on activity levels and interactions are self-reported. Additionally, synthetic demographic data introduces modeling assumptions that may not reflect the true user population.

## Bias Mitigation (edit)
Bias is addressed by focusing on aggregate patterns rather than individual predictions and by clearly distinguishing between observed data and synthetic attributes. Results are interpreted with the understanding that demographic features are simulated rather than observed.

## Rationale for Decisions 
The MovieLens dataset was chosen for its size and relational structure. Synthetic demographic features were added to explore their potential impact on recommendation performance. This introduces additional uncertainty, as demographic attributes are simulated, but allows the project to test whether such information could improve recommendations in practice.

---

# Metadata

## Schema (tba)
(See ER diagram in /docs/erd.png)

## Data Tables (stc)

| Table Name | Description | Link |
|---|---|---|
| ratings | User ratings of movies | ./data/ratings.csv |
| movies | Movie metadata | ./data/movies.csv |
| tags | User-generated tags | ./data/tags.csv |
| links | External movie identifiers | ./data/links.csv |
| users_demo | Synthetic demographic data | ./data/users_demo.csv |

---

# Pipeline (stc)

The pipeline loads data into DuckDB, joins relational tables, and implements a recommendation model to evaluate performance. It includes:

- Data loading
- Query preparation
- Model implementation
- Result visualization

---

# License
This project uses the MIT License.