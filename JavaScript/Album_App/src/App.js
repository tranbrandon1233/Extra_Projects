import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams } from 'react-router-dom';

function Song({ name, genre, duration }) {
  return (
    <div>
      <h3>{name}</h3>
      <p>Genre: {genre}</p>
      <p>Duration: {duration} minutes</p>
    </div>
  );
}

function SongPage({ match }) {
  const { songName } = useParams();
  // Assuming you have a way to fetch song data by name
  const song = { name: songName, genre: 'Pop', duration: 3 };

  return (
    <div>
      <h1>{song.name}</h1>
      <Song name={song.name} genre={song.genre} duration={song.duration} />
    </div>
  );
}

function Album() {
  const [songs, setSongs] = useState([
    { name: 'Song 1', genre: 'Pop', duration: 2 },
    { name: 'Song 2', genre: 'Rock', duration: 4 },
    { name: 'Song 3', genre: 'Jazz', duration: 3 },
  ]);

  const sortSongsByDuration = () => {
    const sortedSongs = [...songs].sort((a, b) => b.duration - a.duration);
    setSongs(sortedSongs);
  };

  return (
    <div>
      <h1>Album Name</h1>
      <button onClick={sortSongsByDuration}>Sort by Duration</button>
      <p>Total Songs: {songs.length}</p>
      <ul>
        {songs.map((song) => (
          <li key={song.name}>
            <Link to={`/song/${song.name}`}>
              <Song name={song.name} genre={song.genre} duration={song.duration} />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Album />} />
        <Route path="/song/:songName" element={<SongPage />} />
      </Routes>
    </Router>
  );
}

export default App;