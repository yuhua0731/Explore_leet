/**
 * 抽象工厂Typescript 实现
 */
interface Dialog {
  showModal();
}
interface Checkbox {
  checklist();
}
interface Button {
  click();
}
abstract class UIFactory {
  abstract createDialog(): Dialog;
  abstract createCheckbox(): Checkbox;
  abstract createButton(): Button;
}

class WinButton implements Button {
  click() {
    console.log('show windows button');
  }
}

class WinCheckBox implements Checkbox {
  checklist() {
    console.log('show windows checkbox');
  }
}

class WinDialog implements Dialog {
  showModal() {
    console.log('show windows dialog');
  }
}

class Win_UIFactory extends UIFactory {
  createButton(): Button {
    return new WinButton();
  }
  createCheckbox(): Checkbox {
    return new WinCheckBox();
  }
  createDialog(): Dialog {
    return new WinDialog();
  }
}

class LinuxButton implements Button {
  click() {
    console.log('show linux button');
  }
}

class LinuxCheckbox implements Checkbox {
  checklist() {
    console.log('show linux checkbox');
  }
}
class LinuxDialog implements Dialog {
  showModal() {
    console.log('show linux dialog');
  }
}

class Linux_UIFactory extends UIFactory {
  createButton(): Button {
    return new LinuxButton();
  }
  createCheckbox(): Checkbox {
    return new LinuxCheckbox();
  }
  createDialog(): Dialog {
    return new LinuxDialog();
  }
}
interface UI {
  button: Button;
  dialog: Dialog;
  checkbox: Checkbox;
}

function app(ui: UI) {
  ui.button.click();
  ui.dialog.showModal();
  ui.checkbox.checklist();
}

function mainApp() {
  const winUI = new Win_UIFactory();
  const button_win: Button = winUI.createButton();
  const dialog_win: Dialog = winUI.createDialog();
  const checkbox_win: Checkbox = winUI.createCheckbox();

  app({ button: button_win, dialog: dialog_win, checkbox: checkbox_win });

  const linuxUI = new Linux_UIFactory();
  const button_linux: Button = linuxUI.createButton();
  const checkbox_linux: Checkbox = linuxUI.createCheckbox();
  const dialog_linux: Dialog = linuxUI.createDialog();

  app({ button: button_linux, dialog: dialog_linux, checkbox: checkbox_linux });
}

mainApp();
