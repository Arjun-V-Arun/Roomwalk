# How It Works

Written to be explained out loud. Each section is one question someone might ask, answered in the order the data flows.

## What Is a Gaussian Splat, and Why Use One for Rooms?

A Gaussian splat scene is a few hundred thousand to a few million small, soft, semi-transparent blobs. Each blob has a position, a shape (three scales and a rotation, together a 3D covariance), an opacity and a colour. The colour can change slightly with viewing angle, stored as spherical harmonic coefficients.

To draw a frame, every blob is projected onto the screen as a 2D ellipse and the ellipses are blended together. Photographs of the real room teach the blobs where to sit and what colour to be.

Why not the alternatives:

- **Photos** are what scammers already fake, and they hide everything outside the frame.
- **A 360 photo** shows one spot. You can't step forward to look at the corner behind the almirah.
- **A textured mesh** needs clean geometry. Rooms are full of thin, shiny and messy things (grills, curtains, bedsheets) that meshes reconstruct badly. Splats handle them because they never need a surface.

## How Does a Video Become a Splat?

**1. Capture check** (`capture/roomwalk/quality.py`). Six frames a second are measured for sharpness (variance of the Laplacian), brightness, blown-out pixels, trackable corners (FAST with an absolute contrast threshold) and camera speed (median optical flow). These predict how reconstruction will go, in seconds instead of after half an hour of training.

One bug worth telling: the first version counted corners with `goodFeaturesToTrack`, whose quality threshold is relative to the strongest corner in the frame. On a blank wall it found 400 "corners" made of sensor noise. The test with a plain synthetic wall caught it.

A known limit: variance of the Laplacian measures high-frequency detail, so it cannot tell a blurry frame from a sharp frame of a blank wall. Both score low. For frame selection that is harmless, because neither helps feature matching, but in a very plain room the blur percentage over-reports.

**2. Frame selection** (`select.py`). The video is cut into 150 equal time windows and the sharpest frame in each is kept. Every-Nth-frame sampling keeps whatever blur landed on that frame; this keeps even coverage without the blur. Windows whose best frame is still blurry are dropped. Frames showing the freshness code are skipped.

**3. Structure from motion** (`sfm.py`, COLMAP through pycolmap). SIFT finds distinctive points in each frame and describes them in a way that survives changes in scale and rotation. Points are matched between frames, and RANSAC throws out matches that don't fit a consistent two-view geometry. Then incremental mapping starts from one good pair, triangulates 3D points, adds each new frame by solving for the pose that explains its matched points (PnP), and repeatedly runs bundle adjustment, which nudges every camera and point together to minimise reprojection error.

Room scans are loops, so the end of the video sees the same wall as the start. Sequential matching only compares nearby frames and never notices the loop closing, so the two ends drift apart. Up to 160 frames, the pipeline compares every pair. Above that it uses quadratic overlap (frame i against i+1, i+2, i+4, i+8 and so on).

**4. Alignment** (`align.py`). COLMAP's answer has no gravity and no units. Three observations fix that from the camera poses alone:

- People don't roll their phones sideways while filming, so each camera's right edge is roughly horizontal. Gravity is the direction most perpendicular to all of them: the smallest-eigenvalue eigenvector of the sum of their outer products. Tilting the phone up or down doesn't move its right edge, so pitch can't bias this. The first version averaged camera up vectors instead; in a test walk that looks down at some walls and level at others, that average is off by 11 degrees, while the right-edge method is within half a degree.
- People film at roughly eye level. If the floor was filmed, camera height above it in model units, divided by 1.45 m, gives the scale.
- The viewer should start facing where the video started, standing in the middle of the walk.

The floor step is where the end-to-end test caught real bugs. The first version took the lowest 2% of sparse points as the floor. In a test walk where the phone stayed nearly level, only 0.1% of points were on the floor, so the "floor" was really the bottom of the walls, and the room came out 1.9 times too big, with no warning. Structure from motion itself was fine: the cameras matched the true path to within a millimetre after a similarity fit.

The second attempt only counted points under the walked area, to exclude walls. That was wrong too: from the middle of a room, the floor you can see is near the walls, not under your feet, so it threw the real floor away.

What actually separates a floor from walls is the height histogram. Walls put points at every height, so their histogram is smooth. A floor is a horizontal surface, so it shows up as a distinct peak: the first well-populated bin from the bottom that is at least 3 times denser than the bins just above it. A bed top is also a peak, so the surface must also span at least 60% of the room in both directions. If no floor qualifies, the ceiling is used, assuming a 2.9 m room height. If neither, a typical room width is assumed and the result is flagged as low confidence, with advice to film the floor.

Final result on the test walk: every frame placed, gravity within 0.01 degrees, scale within 0.5%, camera height within a millimetre. The scan guide now asks people to keep the floor in view during the middle loop.

A second lesson from the same test: a floor of bare tiles gives SIFT almost nothing it can match between views, because the grout grid repeats and plain tile has no texture. Even with the phone pointed at the floor, only 0.2% of points landed there. Lived-in floors have slippers, bags and stains that match well. Very bare, glossy floors are a real risk, which is why the fallback and the low-confidence flag exist.

**5. Training** (`train.py`, Brush). Blobs start at the sparse points. Each step renders the blobs from one training camera, compares the image with the real frame (L1 plus a structural similarity term), and pushes gradients back into every blob's parameters. Blobs where the error stays high are split or cloned (densification); blobs that fade out are removed. The pipeline keeps spherical harmonics at degree 1 (4 coefficients per colour channel instead of 16), because rooms have little shine and it cuts file size.

**6. Packing** (`convert.py`). splat-transform removes broken blobs and floaters, then writes:

- **SOG**: attributes quantised, Morton-ordered and stored as lossless WebP images. The demo room goes from 26.9 MB as PLY to 2.4 MB.
- **RAD** (optional): Spark's level-of-detail tree, split into roughly 1 MB chunks.

## Why Do Splats Need Sorting?

They are semi-transparent, and blending semi-transparent things is order-dependent: red over blue is not blue over red. A depth buffer can't help, because it throws away whatever is behind the nearest surface, and here the things behind still show through. So splats are sorted by distance and blended back to front, and sorted again whenever the camera moves enough to change the order.

Spark sorts in a web worker so the page stays responsive, and keeps drawing with the previous order until the new one arrives. It sorts by radial distance from the camera rather than by depth along the view direction. Radial distance does not change when you only rotate, and turning on the spot is most of what people do inside a room, so rotating doesn't trigger visible popping.

## How Does Streaming Decide What Loads Next?

The RAD file is a tree. Each parent blob is a merged approximation of its children, so the root levels are a blurry version of the whole room and deeper levels add detail.

Every frame, Spark walks the tree from the root and opens a node only if its blobs would cover more than about one pixel on screen, until the frame's splat budget is spent. Things far away or behind you stay coarse. Detail goes where you are looking.

The tree is stored in chunks. The first chunk holds the coarse levels, so the room appears once the small header and that first chunk arrive. Further chunks are fetched as the walk needs them. On the GPU, a fixed pool of splat pages works like virtual memory: chunks are paged in, and ones not needed recently are evicted.

## What Does the Viewer Add on Top of Spark?

- **Render on demand.** Frames are drawn only when the camera moves, an animation runs, or Spark reports new work (a sort finished, a chunk arrived). Measured: zero to three frames in three idle seconds, instead of 180.
- **Device tiers.** A starting splat budget and pixel ratio from the GPU name, core count and memory. Budget phones start at 1x pixels and 45% of Spark's budget.
- **Adaptive quality.** While you interact, the median frame time is checked every 1.5 s. Below 26 fps it lowers the pixel ratio first, then the splat budget. Above 52 fps for three checks in a row it raises them back. Pixel ratio goes first because splat cost scales with pixels covered.
- **Honest loading.** The room's real photo is shown immediately and fades out when the 3D is actually on screen. "On screen" is measured by reading back pixels after rendering, not by when the file finished downloading.
- **Hotspots and double-tap to walk.** Fly-to animations between saved views; double-tap raycasts into the splats and walks you toward the spot, stopping 0.7 m short.

## What Does the Freshness Code Prove?

The platform signs `RW1:<room>:<issue time>` with HMAC-SHA256 and a secret key, and shows it as a QR code. The tenant films it first. The check decodes it, verifies the signature in constant time, confirms the room id matches, and checks the phone's recording timestamp falls within 48 hours after issue.

So the video was made recently, by someone we sent the code to. It does not prove the room is at the listed address, and the site says so in plain words. In a real product, codes would be issued by a server rather than a laptop, so landlords couldn't see or reuse the key.

## Questions to Expect

**Why not just stream video of the room?** Video plays a path someone else chose. The point is that the renter decides where to look.

**What breaks it?** Plain walls (nothing to match), mirrors and windows (the reconstruction puts a second room behind them), moving things (fans, people), and dim light (blur and noise). The capture check catches most of these before training.

**Why is first view faster with RAD when RAD is bigger?** SOG must arrive completely and then be turned into a level-of-detail tree on the phone. RAD arrives already organised as a tree, and the coarse levels are at the front, so the room appears after a fraction of the download. The trade is more total bytes if you look at everything.

**What would you build next?** A small Go service that issues codes and receives uploads, so tenants can submit scans from their phone. Real phone measurements across budget Android devices. Compressing RAD chunks further for metered data.
