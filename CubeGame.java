
import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;
import java.util.Random;
import javax.swing.*;

public class CubeGame extends JFrame {
    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;

    private int score = 0;
    private boolean gameOver = false;
    private boolean win = false;

    public CubeGame() {
        setSize(WIDTH, HEIGHT);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setVisible(true);

        JPanel panel = new JPanel() {
            @Override
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                if (gameOver && !win) {
                    g2d.drawString("Game Over! Your score is: " + score, 300, 300);
                } else if (score == 5) {
                    win = true;
                    g2d.drawString("You Win! Your score is: " + score, 300, 300);
                }

                else {
                    g2d.drawString("Score: " + score, 10, 40);
                }
                for (Cube cube : cubes) {
                    if (cube == null) {
                        continue;
                    }
                    if (cube.isVisible()) {
                        cube.draw(g2d);
                    }
                }
            }
        };
        add(panel);

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                for (int i = 0; i < cubes.length; i++) {
                    if (cubes[i] == null) {
                        continue;
                    }
                    if (cubes[i].contains(e.getX(), e.getY())) {
                        if (cubes[i].isRed()) {
                            gameOver = true;
                            repaint();
                        } else {
                            score++;
                            if (score >= 5) {
                                win = true;
                                repaint();
                                break;
                            }
                            cubes[i] = null;
                            if (Arrays.stream(cubes).noneMatch(Cube::isRed)) {
                                gameOver = true;
                                repaint();
                            }
                        }
                        break;
                    }
                }
            }
        });

        Timer timer = new Timer(1000, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (!gameOver && !win) {
                    for (Cube cube : cubes) {
                        if (cube == null) {
                            continue;
                        }
                        if (cube.isVisible()) {
                            cube.setVisible(false);
                        } else {
                            cube.setVisible(true);
                            cube.move();
                        }
                    }
                    repaint();
                } else {
                    for (int i = 0; i < cubes.length; i++) {
                        cubes[i] = null;

                    }
                }
            }
        });
        timer.start();
    }

    private class Cube {
        private int x, y, w, h;
        private Color color;
        private boolean isVisible;

        public Cube(int x, int y, int w, int h, Color color) {
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.color = color;
            this.isVisible = true;
        }

        public boolean isRed() {
            return color.equals(Color.RED);
        }

        public void move() {
            Random random = new Random();
            x = random.nextInt(WIDTH - w);
            y = random.nextInt(HEIGHT - h);
        }

        public void draw(Graphics2D g2d) {
            g2d.setColor(color);
            g2d.fillRect(x, y, w, h);
        }

        public boolean contains(int x, int y) {
            return this.x <= x && x <= this.x + w && this.y <= y && y <= this.y + h;
        }

        public boolean isVisible() {
            return isVisible;
        }

        public void setVisible(boolean isVisible) {
            this.isVisible = isVisible;
        }
    }

    private Cube[] cubes = new Cube[10];

    {
        Random random = new Random();
        if (!gameOver && !win) {
            for (int i = 0; i < cubes.length; i++) {

                cubes[i] = new Cube(random.nextInt(WIDTH - 50), random.nextInt(HEIGHT - 50), 50, 50,
                        i % 2 == 0 ? Color.RED : Color.GREEN);
            }
        }

    }

    public static void main(String[] args) {
        new CubeGame();
    }
}
