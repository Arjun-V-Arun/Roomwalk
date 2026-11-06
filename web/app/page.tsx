import Link from "next/link";
import { Trust } from "@/components/Stamp";
import { roomUrl, rupees, shortDate } from "@/lib/format";
import { listRooms } from "@/lib/rooms";

const REPO = "https://github.com/Arjun-V-Arun/roomwalk";

export default function Home() {
  const rooms = listRooms();
  return (
    <main className="page">
      <header className="masthead">
        <Link href="/" className="wordmark">
          RoomWalk
        </Link>
        <a className="quiet" href={`${REPO}/blob/main/docs/SCAN_GUIDE.md`}>
          List your room
        </a>
      </header>

      <section className="lede">
        <h1 className="display">Walk through the room before you pay the deposit.</h1>
        <p>
          Each room here was filmed on a phone by someone living in it, then rebuilt in 3D. Open one and look at the
          corners, the window and the state of the floor yourself.
        </p>
      </section>

      {rooms.length === 0 ? (
        <p className="empty">No rooms yet. Publish one with the capture tool and rebuild the site.</p>
      ) : (
        <ul className="rooms">
          {rooms.map((r) => (
            <li key={r.id}>
              <Link href={`/r/${r.id}/`} className="room-card">
                <div className="room-photo">
                  {r.poster && (
                    // Plain img: the site is a static export, so there is no image server.
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={roomUrl(r.id, r.poster)} alt="" loading="lazy" />
                  )}
                  {r.verified && !r.synthetic && <Trust verified scannedAt={r.scannedAt} />}
                </div>
                <div className="room-body">
                  {r.rent ? (
                    <div className="room-price num">
                      {rupees(r.rent)} <span>a month</span>
                    </div>
                  ) : null}
                  <h2 className="room-title display">{r.title}</h2>
                  <div className="room-meta">
                    {r.area && <span>{r.area}</span>}
                    {r.scannedAt && !r.synthetic && <span>Filmed {shortDate(r.scannedAt)}</span>}
                    {(r.synthetic || !r.verified) && <Trust verified={r.verified} synthetic={r.synthetic} />}
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}

      <section className="how" aria-labelledby="how-title">
        <div>
          <h2 id="how-title" className="display">
            How a room gets here
          </h2>
          <ol className="steps">
            <li>
              <div>
                <strong>We send the tenant a one-time code.</strong>
                <p>They open it on a second screen and film it before anything else.</p>
              </div>
            </li>
            <li>
              <div>
                <strong>They film the room for about a minute.</strong>
                <p>Lights on, a slow loop around the walls, then a second loop from the middle.</p>
              </div>
            </li>
            <li>
              <div>
                <strong>We check the video and rebuild the room.</strong>
                <p>Blurry, dark or half-covered videos are sent back with what to fix. Good ones become 3D.</p>
              </div>
            </li>
            <li>
              <div>
                <strong>You open the link and look around.</strong>
                <p>It works in the phone browser. Nothing to install.</p>
              </div>
            </li>
          </ol>
        </div>
        <aside className="proves">
          <span className="stamp">
            Code checked
            <small>Filmed 9 Nov 2026</small>
          </span>
          <p>
            This stamp means our code appears in the video and the phone recorded it within two days of us sending
            it. So the room looked like this recently, filmed by someone we were in touch with.
          </p>
          <p style={{ marginTop: "0.7rem" }}>
            It does not prove the room is at the listed address. Visit, or video-call from inside, before you pay
            anything.
          </p>
        </aside>
      </section>

      <footer className="foot">
        <a href={REPO}>Source code</a>. Rooms are Gaussian splats rendered with Spark and three.js.
      </footer>
    </main>
  );
}
