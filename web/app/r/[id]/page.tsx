import type { Metadata } from "next";
import { RoomPage } from "@/components/RoomPage";
import { roomUrl, rupees } from "@/lib/format";
import { getRoom, listRooms } from "@/lib/rooms";

export const dynamicParams = false;

export function generateStaticParams() {
  return listRooms().map((r) => ({ id: r.id }));
}

type Props = { params: Promise<{ id: string }> };

// Links get shared on WhatsApp, so each room needs its own preview card.
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  const m = getRoom(id);
  const price = rupees(m.rent);
  const description = [price && `${price} a month`, m.area, "Walk through it in 3D on your phone."]
    .filter(Boolean)
    .join(". ");
  return {
    title: m.title,
    description,
    openGraph: {
      title: m.title,
      description,
      type: "website",
      images: m.poster ? [{ url: roomUrl(m.id, m.poster), width: 1200 }] : undefined,
    },
  };
}

export default async function Room({ params }: Props) {
  const { id } = await params;
  return <RoomPage manifest={getRoom(id)} />;
}
