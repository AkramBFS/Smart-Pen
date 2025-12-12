

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;
import java.util.concurrent.Semaphore;

public class philosopherSolution2  extends JFrame implements Runnable{
    int id;

    static boolean[] chopstickAvailable = new boolean[5];
    static Semaphore[] semPhilosopher = new Semaphore[5];


    JLabel state;
    JLabel rightChopstick;
    JLabel leftChopstick;

    final int WINDOW_HEIGHT = 100;
    final int WINDOW_WIDTH = 300;

    static {
        Arrays.fill(chopstickAvailable, true);
        for (int i = 0; i < semPhilosopher.length; i++) {
            semPhilosopher[i] = new Semaphore(0, true);
        }
    }

    public philosopherSolution2(int id) throws HeadlessException {
        this.id = id;

        this.setTitle("Philosopher" + id);
        this.setSize(WINDOW_WIDTH, WINDOW_HEIGHT);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.setLocationRelativeTo(null);
        this.setVisible(true);

        state = new JLabel();
        rightChopstick = new JLabel();
        leftChopstick = new JLabel();

        JPanel jPanel = new JPanel();
        jPanel.add(state);
        jPanel.add(rightChopstick);
        jPanel.add(leftChopstick);
        this.add(jPanel);
    }

    private void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void run() {
        while (true) {
            
        }
    }
    private void eat() {
        state.setText("Eating");
        sleep(5000);
    }

    private void think() {
        state.setText("Thinking");
       sleep(10000);
    }

    private void takeChopsticks() {
        
        rightChopstick.setText(String.format("Has fork %d", id));
        leftChopstick.setText(String.format("Has fork %d", (id + 1) % 5));
    }


    private void putChopsticks() {
       
        rightChopstick.setText("");
        leftChopstick.setText("");
        
    }
}