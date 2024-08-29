import pyautogui

class Screen_Mouse:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def move_to(self,x,y):
        pyautogui.moveTo(x, y)


    def move(self,x,y):
        pyautogui.move(x, y)

    def drag_to(self,x,y,time):
        pyautogui.dragTo(x, y,  duration=time)

    def drag(self,x,y,time):
        pyautogui.drag(x, y, duration=time)
    
    def click(self):
        pyautogui.click()

    def double_click(self):
        pyautogui.doubleClick()

    def right_click(self):
        pyautogui.rightClick()

    def get_position(self):
        current_mouse_x, current_mouse_y = pyautogui.position()
        return current_mouse_x, current_mouse_y

def main():
    mouse  = Screen_Mouse()
    mouse.move_to(50,50)

if __name__ == "__main__":
    main()