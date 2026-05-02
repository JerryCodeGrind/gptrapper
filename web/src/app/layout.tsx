import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Found-Sound Song Maker',
  description: 'Make a song using your own noises',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
