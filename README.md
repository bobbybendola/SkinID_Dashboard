# Skin Disease Identifier
SkinID was inspired by the gap between how easily people can capture images of skin concerns and how difficult it is to extract consistent, structured measurements from those images. 
Skin conditions are often evaluated visually, yet the data collected is frequently unstandardized, which limits its usefulness.

The application allows users to upload an image of a skin concern, visually mark a deformity using a bounding box, and automatically calculate its height and width in pixels.
In addition, the interface collects structured user metadata—such as age, sex, and anatomical location—and submits the consolidated data to a backend API using multipart form submission.
