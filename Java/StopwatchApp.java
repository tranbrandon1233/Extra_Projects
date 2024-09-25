import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

public class StopwatchApp extends JFrame {
    private JLabel timeLabel;
    private JButton startButton;
    private JButton lapButton;
    private JButton pauseButton;
    private JButton resetButton;
    private JTextArea lapTextArea;
    private JPanel lapPanel;
    private Timer timer;
    private int seconds = 0;
    private int minutes = 0;
    private int hours = 0;
    private List<String> lapTimes = new ArrayList<>();

    public StopwatchApp() {
        super("Stopwatch");
        setLayout(new BorderLayout());

        // Create the stopwatch panel
        JPanel stopwatchPanel = new JPanel();
        stopwatchPanel.setLayout(new FlowLayout());
        timeLabel = new JLabel("00:00:00");
        startButton = new JButton("Start");
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                startStopwatch();
            }
        });
        pauseButton = new JButton("Pause");
        pauseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                pauseStopwatch();
            }
        });
        resetButton = new JButton("Reset");
        resetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                resetStopwatch();
            }
        });
        stopwatchPanel.add(timeLabel);
        stopwatchPanel.add(startButton);
        stopwatchPanel.add(pauseButton);
        stopwatchPanel.add(resetButton);

        // Create the lap panel
        lapPanel = new JPanel();
        lapPanel.setLayout(new FlowLayout());
        lapTextArea = new JTextArea();
        lapTextArea.setEditable(false);
        lapTextArea.setBackground(Color.WHITE);
        lapButton = new JButton("Lap");
        lapButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                recordLap();
            }
        });
        JScrollPane scrollPane = new JScrollPane(lapTextArea);
        lapPanel.add(scrollPane);
        lapPanel.add(lapButton);

        // Add the stopwatch and lap panels to the main frame
        add(stopwatchPanel, BorderLayout.NORTH);
        add(lapPanel, BorderLayout.SOUTH);

        // Set the size and visibility of the frame
        setSize(300, 200);
        setVisible(true);
    }

    private void startStopwatch() {
        timer = new Timer(1000, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                updateTimeLabel();
            }
        });
        timer.start();
    }

    private void pauseStopwatch() {
        timer.stop();
    }

    private void resetStopwatch() {
        timer.stop();
        seconds = 0;
        minutes = 0;
        hours = 0;
        timeLabel.setText("00:00:00");
        lapTimes.clear();
        lapTextArea.setText("");
    }

    private void recordLap() {
        lapTimes.add(String.format("Lap %d: %02d:%02d:%02d", lapTimes.size() + 1, hours, minutes, seconds));
        lapTextArea.setText(String.join("\n", lapTimes));
    }

    private void updateTimeLabel() {
        seconds++;
        if (seconds >= 60) {
            minutes++;
            seconds = 0;
        }
        if (minutes >= 60) {
            hours++;
            minutes = 0;
        }
        timeLabel.setText(String.format("%02d:%02d:%02d", hours, minutes, seconds));
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new StopwatchApp();
            }
        });
    }
}