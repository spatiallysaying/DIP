Runway Category Detection 
Team-Delta One
Github link (see instructions for this in the next step)
Team Members -DURGA PRASAD
Main goal(s) of the project 
 Detect Runway Category by image processing
Problem definition (What is the problem? How things will be done ?)

 Background
 ----------
 Airport pavement markings and signs provide information that is useful to a pilot during takeoff, landing, and taxiing.
 In this study , we are considering Runway threshold markings which is one such type .A threshold marking helps identify 
 the beginning of the runway that is available for landing.They consist of longitudinal stripes of uniform dimensions 
 disposed symmetrically about the runway centerline and the number of stripes is related to the runway width as defined 
 in ICAO standard. 
 
 Problem 
 -------
 Conventionally,Runway width is known through  published airport charts or GIS (Geographic Information System) maps. 
 Manual digitization is used for extracting these maps. The issue is to identify the width of runway automatically.
 
 Approach
 --------
 Locate the threshold markings and  count the number of stripes automatically and then estimate runway width per ICAO standard. 
 

Results of the project (What will be done? What is the expected final result ?)
 Identify the threshold-which I accomplished in my thesis work using ML.
 Use Image processing , to extract boundaries of the stripes.
 Make the extraction invariant of pavement type and wear and tear  .
 Count the number of stripes.
 Estimate width and hence identify the runway category.
 
 
What are the project milestones and expected timeline ?
 
 Step1:
 -----
 Experiment with below papers to detect edges
 31 An Algorithm for Detecting Lines Based on Primitive Connection 
 24 Road Detection by Using a Generalized Hough Transform on Indian Roads 
 37 Bilateral image inpainting 
 Step2:
 -----
 Pavement Invariance
 
 Step3:
 -----
 Boundary Detection and stripe counting
 
 Step4:
 -----
 Width detection and Runway type categorization
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 