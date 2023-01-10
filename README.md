<h1>
  <img src="https://govhack-hackerspace.s3.ap-southeast-2.amazonaws.com/kXLr9szHrSj3R2qQSLiFVhH5?response-content-disposition=inline%3B%20filename%3D%22Insolvency.png%22%3B%20filename%2A%3DUTF-8%27%27Insolvency.png&response-content-type=image%2Fpng&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAI7HPM2TPQIOGV6OA%2F20230110%2Fap-southeast-2%2Fs3%2Faws4_request&X-Amz-Date=20230110T014926Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=00493e91a9538533cbf7ed00ebb1cae1995b52947aadd353314257aeec0b9082" height="36" valign="bottom" /> inSOLVEncy_2018_GovHack
</h1>

## Project Description
Using ML to identify which individuals will commit insolvency by creating a compliance risk model and visualizing the results.

[Our GovHack Project Page](https://hackerspace.govhack.org/projects/insolvency_113)

[Watch our Project Summary Video](https://www.youtube.com/watch?v=8L1BdYll2TY)

[Bounty Winner: Best use of Gold Coast Data](https://www.govhack.org/2018-winners)

## Data Story
Our project inSOLVEnt takes a multifaceted approach to what is a multifaceted problem by creating not only a risk model for addressing non-compliance to personal insolvency, but visualisations and infographics addressing the common factors leading to negative insolvent outcomes.

We utilised the non-compliance personal insolvency data to first identify cases of non-compliance versus compliance. We streamlined this data using other data sources such as Regional Statistics, the ATO GovHack 2018 statistics, and ANZSCO occupation and regional classifications.

Once we had a clean data set we ran through tensa flows to identify a model. We tried neural networks first, which were overfitting and not generalising in our tests. We decided to simplify using linear regressions which worked well. Out of 250,000 records we misidentified 5. This is an incredible accuracy result.
When training our model, we first separated all compliance and all non-compliance. Each where then randomly split using an 80% training, and 20% validation split. As non-compliance events were the minority, this method was to ensure that our training subsets were balanced.

We further delved into these results by isolating Gold Coast data by utilising AS3 data sets. We retrained our model in the same method. Our validation results for the Gold Coast data was 100%. The Gold Coast data was consistent with the National model, reinforcing the robustness of our solution.
We retained both models using mean absolute error, rather than mean squared error, as mean squared error amplifies outliers.

Our model is able to predict non-compliance events to a high degree of accuracy. This risk model can be used by regulatory bodies to target audit and compliance services, and individuals and corporate entities to self-identify their compliance risk.
