package com.akuvox.speech;

import android.app.Application;

import com.hjq.toast.Toaster;

public final class AppApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();
        Toaster.init(this);
    }
}