// src/__test__/LandingPage.test.jsx
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import LandingPage from '../pages/LandingPage';

beforeAll(() => {
  global.IntersectionObserver = class IntersectionObserver {
    constructor(callback) {
      this.callback = callback;
    }
    observe() {
      this.callback([{ isIntersecting: true }]);
    }
    unobserve() {}
    disconnect() {}
  };
});

describe('Landing Page - Tests Simples', () => {
  test('le composant se rend sans erreur', () => {
    render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>
    );
    expect(true).toBe(true);
  });

  test('affiche le titre principal', () => {
    render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>
    );
    
    // Vérifie simplement que le texte existe
    expect(screen.getByText(/SpiriComp/i)).toBeTruthy();
  });

  test('affiche au moins un bouton de lancement', () => {
    render(
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>
    );
    
    // Vérifie qu'il y a des boutons avec le texte
    const buttons = screen.getAllByText(/landing.launch|launch/i);
    expect(buttons.length).toBeGreaterThan(0);
  });
});