# KreveraProject
At a high level these scripts construct the convex hull of the given mesh, places and rotates the projection of a synthetic burn mark on the hull, and then uses replicator to automate the process of generating the data.  To run a script, open it in Omniverse Code Beta 3.1.1 Script Editor, edit the texture_path to match your path to black_circle.png and then run the script to setup the scene so that you can preview it with replicator.  

The torus demo below highlights some of the shortcomings of this technique: its possible that a chosen projection position is not in front of the torus; if the projection direction happens to intersect with object twice the burn mark will be visible in both locations; its not possible to directly project on to a point that is not on the convex hull (e.g. the inner ring). Some of these problems are due to my technique, but the projection material is currently flawed for our pupose and the requires modifying Omniverse to resolve the issue. Also he synthetic burn mark is a black circle that is randomly scaled and it is segmentable; modification to Omniverse is required to gain further control over the texture. 


![](./burn_torus_demo.gif)
