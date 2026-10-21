import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { RouterModule, Routes } from '@angular/router';

import { AppComponent } from './app.component';
import { ChatComponent } from './chat/chat.component';
import { AdminComponent } from './admin/admin.component';

const routes: Routes = [
  { path: '', component: ChatComponent },
  { path: 'admin', component: AdminComponent },
  { path: '**', redirectTo: '' },
];

@NgModule({
  declarations: [AppComponent, ChatComponent, AdminComponent],
  imports: [BrowserModule, FormsModule, HttpClientModule, RouterModule.forRoot(routes)],
  bootstrap: [AppComponent]
})
export class AppModule {}
