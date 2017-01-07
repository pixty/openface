# openface
Custom Openface API 

to build an image

- docker build -t openface_api . 

to run a container

- docker run -d -p 5000:5000 openface_api

Test using a browser open any link below
http://localhost:5000/getAllFaceBoundingBoxes
http://localhost:5000/getFacesMeta
http://localhost:5000/getAllFacesMeta
http://localhost:5000/getAllFaces

Fill in a form and press the "Upload" button

