# DS 4320 Project 1: Do User Demographics Improve Movie Recommendation Systems?

This repository contains a data pipeline and analysis that evaluate whether user demographic information improves recommendation systems. It includes scripts to construct a relational database from a large-scale user–movie interaction dataset, generate synthetic demographic attributes, and prepare data for modeling. The analysis compares a baseline model using only behavioral data with an enhanced model that incorporates demographics to predict user movie ratings on a 5-star scale. Results show that demographic features provide virtually no improvement in predictive performance, indicating that user behavior alone is sufficient for generating effective recommendations.

Name: Liam Ward<br>
NetID: vhk7vr

DOI: tba<br>
[Press Release](./press_release.md)<br>
[Data Folder](./data/) this will be onedrive<br>
[Pipeline](./pipeline/)<br>
[MIT License](./LICENSE)<br>


## Problem Definition

**General Problem:**    
Recommending content

**Refined Problem:**    
Determine whether incorporating user demographic information (age group and sex) improves the accuracy of predicting user movie ratings, and therefore leads to better content recommendations, compared to using behavioral data alone.

**Rationale for refinement:**   
The general problem of recommending content is broad and involves many possible approaches, including improving algorithms, collecting new data, or refining existing features. This project narrows the focus to a specific and practical question: whether demographic data improves recommendation accuracy. By isolating the impact of demographic features, the project enables a clear, testable evaluation of whether this type of data is worth incorporating into recommendation systems.

**Motivation**  
Content recommendation systems are widely used in platforms such as streaming services, where decisions about what data to collect directly affect system complexity, cost, and user privacy. Demographic information is often assumed to improve personalization, but collecting and using it may not always be necessary. This project is motivated by the need to determine whether demographic data provides meaningful improvements in recommendation accuracy. If it does not, systems can remain simpler, more efficient, and less reliant on sensitive user information.

**[Headline]()**
